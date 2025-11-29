using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Runtime.Serialization.Formatters;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;

namespace Recon3D
{
    internal struct PMap
    {
        internal double probability;
        internal int relative_pointX;
    }

    internal class Poser
    {
        internal Point2f[]? points1;
        internal Point2f[]? points2;
        internal Point3f[]? points3D;
        internal Vec3b[]? colors;
        internal Mat? R, t;
        internal Mat? cm;
        private Point2f[] base_points1;
        private Point2f[] base_points2;
        private Point3f[] base_points3D;
        private Vec3b[] base_colors;

        /// <summary>
        /// Creates a dense point cloud from the given images
        /// </summary>
        /// <param name="img0">image 0</param>
        /// <param name="img1">image 1</param>
        /// <param name="HFOV">camera field of view</param>
        /// <returns>Poser</returns>
        internal static Poser Create(Mat img0, Mat img1, float HFOV)
        {
            //first we need some base points so let's perform some
            //dense matching
            var pose1 = Poser.FindMatchingBasePoints(img0, img1);

            //swap images and run the same algo
            var pose1r = Poser.FindMatchingBasePoints(img1, img0);

            //add the points to pose1 (they're reversed)
            pose1.AddReversedPoints(pose1r);

            //create a dense 3d point cloud
            pose1.TriangulatePoints(img0, img1, HFOV);
            return pose1;
        }

        /// <summary>
        /// Creates a dense match between 2 images
        /// </summary>
        /// <param name="img1">image 1</param>
        /// <param name="img2">image 2</param>
        /// <returns>Poser</returns>
        static Poser FindMatchingBasePoints(Mat img1, Mat img2)
        {
            var mat1 = img1.Clone();
            var mat2 = img2.Clone();

            //find matching keypoints between the images
            var mtch = Matched.FindMatches(mat1, mat2);

            //filter any noise
            mtch.FilterSimilarSizes(0.6d);

            //calculate the fundamental matrix
            mtch.CalculateFundamentalAndFilter();

            //modify the colors of the image to match
            //the 2nd image, use keypoints as hints
            mat1.NormalizeWith(mat2,
                mtch.keypoints1.Select(s => s.Pt.ToPoint()).ToArray(),
                mtch.keypoints2.Select(s => s.Pt.ToPoint()).ToArray());

            //find all matches using the color balanced images
            mtch = Matched.FindMatches(mat1, mat2);
            mtch.FilterSimilarSizes(0.6d);
            mtch.CalculateFundamentalAndFilter();

            //calculate the homography between the images
            mtch.CalculateH1H2();

            //transform the images to match the homography
            mtch.WarpTransformsWithMasks(5);

            //warp the keypoints according to the homography
            mtch.WarpPointsH1H2();

            //filter any points that don't share the same
            // y value
            mtch.FilterSimilarYValues(1);

            //generate some canny edges so we can tease out
            //any signals
            var can1 = mtch.image1!.AdaptiveCannyExt(21);
            var can2 = mtch.image2!.AdaptiveCannyExt(21);

            can1 = can1.ApplyMask(mtch.mask1!);
            can2 = can2.ApplyMask(mtch.mask2!);

            int sqsize = 14;  //size of the window used in matching between images
            int wiggle = 30;  //the allowable search area in the 2nd image

            double lowerb = 0.15;  //the lower bound when identifying signals
            double upperb = 1 - lowerb;  //the upper bound when identifying signals
            double pvalue = 0.8;  //the lower limit on a match's p-value
            double ttlpts = sqsize * sqsize;

            //map the whole surface in probabilities of successful matches
            PMap[,] pMaps = new PMap[can1.Width, can1.Height];

            // Pre-convert images for template matching (use color match for better fidelity)
            var searchImg = mtch.image2!;
            var templImg = mtch.image1!;

            //get squares that match using template matching
            for (int y = 0; y < can1.Height - sqsize; y += sqsize / 2)
            {
                for (int x1 = 0; x1 < can1.Width - sqsize; x1 += sqsize / 2)
                {
                    double maxp = 0;
                    int sx1 = 0, sx2 = 0;

                    //get the square in the Canny image to check signal
                    var rect1 = new Rect(x1, y, sqsize, sqsize);
                    var canSq1 = can1[rect1];

                    //does it have enough information?
                    var sum1 = canSq1.Sum();
                    var avg1 = sum1.Val0 / ttlpts / 255;

                    //find the square in the 2nd image using MatchTemplate
                    if (avg1 >= lowerb && avg1 <= upperb)
                    {
                        // template from the color-balanced warped image
                        using var template = templImg[rect1];

                        int x2_start = Math.Max(0, x1 - wiggle);
                        int x2_end = Math.Min(can2.Width - sqsize, x1 + wiggle);

                        if (x2_start <= x2_end)
                        {
                            int searchWidth = x2_end - x2_start + sqsize;
                            var searchRect = new Rect(x2_start, y, searchWidth, sqsize);

                            using var searchStrip = searchImg[searchRect];

                            // perform template matching on the strip
                            using var result = new Mat();
                            Cv2.MatchTemplate(searchStrip, template, result, TemplateMatchModes.CCoeffNormed);

                            // find best match
                            Cv2.MinMaxLoc(result, out _, out double maxVal, out _, out Point maxLoc);

                            if (maxVal >= pvalue)
                            {
                                maxp = maxVal;
                                sx1 = x1;
                                sx2 = x2_start + maxLoc.X;
                            }
                        }
                    }

                    if (maxp > 0)
                    {
                        //add to the pMaps
                        for (int szX = 0; szX < sqsize; szX++)
                            for (int szY = 0; szY < sqsize; szY++)
                            {
                                int px = sx1 + szX;
                                int py = y + szY;
                                if (px >= 0 && px < pMaps.GetLength(0) && py >= 0 && py < pMaps.GetLength(1))
                                {
                                    if (maxp > pMaps[px, py].probability)
                                    {
                                        pMaps[px, py].probability = maxp;
                                        pMaps[px, py].relative_pointX = sx2 + szX;
                                    }
                                }
                            }
                    }
                }
            }

            //generate a dense pointcloud from pMaps

            List<Point2f> p1 = new List<Point2f>();
            List<Point2f> p2 = new List<Point2f>();

            for (int x = 0; x < can1.Width; x++)
            {
                for (int y = 0; y < can1.Height; y++)
                {
                    var map = pMaps[x, y];
                    if (map.probability > 0)
                    {
                        var pt1 = new Point2f(x, y);
                        var pt2 = new Point2f(map.relative_pointX, y);

                        p1.Add(pt1);
                        p2.Add(pt2);
                    }
                }
            }

            //unwarp the points
            mtch.keypoints1 = p1.ToArray().ToKeyPoints();
            mtch.keypoints2 = p2.ToArray().ToKeyPoints();

            mtch.UnWarpPointsH1H2();

            var pose = new Poser();
            pose.points1 = mtch.keypoints1.ToPoints2f();
            pose.points2 = mtch.keypoints2.ToPoints2f();

            return pose;
        }

        /// <summary>
        /// Triangulates 3d points from points1 and points2 while
        /// filtering using the fundamental matrix
        /// </summary>
        /// <param name="img1"></param>
        /// <param name="img2"></param>
        /// <param name="HFOV"></param>
        internal void TriangulatePoints(Mat img1, Mat img2, float HFOV)
        {
            var mtch = new Matched(img1.Size());
            mtch.keypoints1 = points1!.ToKeyPoints();
            mtch.keypoints2 = points2!.ToKeyPoints();
            mtch.image1 = img1;
            mtch.image2 = img2;
            mtch.CalculateFundamentalAndFilter();
            cm = mtch.GetCameraMatrix(HFOV);


            Point3f[] ProjectedPoints;
            Triangulation.Triangulate(cm,
                mtch.GetPoints1(),
                mtch.GetPoints2(),
                out R, out t, out ProjectedPoints);

            mtch.ProjectedPoints = ProjectedPoints;
            mtch.SetSourceColors1();

            points3D = mtch.ProjectedPoints;
            colors = mtch.SourceColors;
            points1 = mtch.GetPoints1();
            points2 = mtch.GetPoints2();
        }


        internal void ExportOBJ(string fileName)
        {
            var mtch = new Matched(new Size(100, 100));
            mtch.ProjectedPoints = points3D!;
            mtch.SourceColors = colors!;
            mtch.ExportOBJ(fileName);
        }

        internal void ExportPLY(string fileName)
        {
            var mtch = new Matched(new Size(100, 100));
            mtch.ProjectedPoints = points3D!;
            mtch.SourceColors = colors!;
            mtch.ExportPLY(fileName);
        }

        internal void ExportPCD(string fileName)
        {
            var mtch = new Matched(new Size(100, 100));
            mtch.ProjectedPoints = points3D!;
            mtch.SourceColors = colors!;
            mtch.ExportPCD(fileName);
        }

        internal void ExportPoints(string fileName)
        {
            var mtch = new Matched(new Size(100, 100));
            mtch.ProjectedPoints = points3D!;
            mtch.SourceColors = colors!;
            mtch.ExportPoints(fileName);
        }

        /// <summary>
        /// Performs a logical AND on the points
        /// </summary>
        /// <param name="s2"></param>
        internal void KeepSamePoints(Poser s2)
        {
            var m1 = new Matched(new Size(100, 100));
            var m2 = new Matched(new Size(100, 100));

            m1.keypoints1 = points1!.ToKeyPoints();
            m1.keypoints2 = points2!.ToKeyPoints();
            m1.SourceColors = this.colors!;
            m1.ProjectedPoints = points3D!;

            m2.keypoints1 = s2.points1!.ToKeyPoints();
            m2.keypoints2 = s2.points2!.ToKeyPoints();
            m2.SourceColors = s2.colors!;
            m2.ProjectedPoints = s2.points3D!;

            //force the same number of points
            m1.FilterSharedAbsPoints(m2);

            this.points1 = m1.keypoints1.ToPoints2f();
            this.points2 = m1.keypoints2.ToPoints2f();
            this.colors = m1.SourceColors;
            this.points3D = m1.ProjectedPoints;

            s2.points1 = m2.keypoints1.ToPoints2f();
            s2.points2 = m2.keypoints2.ToPoints2f();
            s2.colors = m2.SourceColors;
            s2.points3D = m2.ProjectedPoints;
        }

        /// <summary>
        /// Generates images for both camera views
        /// </summary>
        /// <param name="file1"></param>
        /// <param name="file2"></param>
        /// <param name="size"></param>
        internal void ReprojectPoints(string file1, string file2, Size size)
        {
            var res1 = new Mat(size, MatType.CV_8UC3);
            res1.SetTo(new Scalar(0));

            var res2 = new Mat(size, MatType.CV_8UC3);
            res2.SetTo(new Scalar(0));

            var distCoeffs = Triangulation.GetDistCoeff();
            var cam1 = Triangulation.ReprojectPoints(points3D!, cm!, R!, t!, distCoeffs, Triangulation.CamNumber.One);
            var cam2 = Triangulation.ReprojectPoints(points3D!, cm!, R!, t!, distCoeffs, Triangulation.CamNumber.Two);

            for (int i = 0; i < cam1.Length; i++)
            {
                var p1 = cam1[i];
                var p2 = cam2[i];
                var co = colors![i];

                if (!res1.InvalidPoint(p1))
                {
                    res1.Set<Vec3b>((int)p1.Y, (int)p1.X, co);
                }
                if (!res2.InvalidPoint(p2))
                {
                    res2.Set<Vec3b>((int)p2.Y, (int)p2.X, co);
                }
            }

            res1.SaveImage(file1);
            res2.SaveImage(file2);
        }

        /// <summary>
        /// Reverses then adds points
        /// </summary>
        /// <param name="pose1r">The Poser to add</param>
        internal void AddReversedPoints(Poser pose1r)
        {
            List<Point2f> p1 = new List<Point2f>();
            List<Point2f> p2 = new List<Point2f>();

            p1.AddRange(points1!);
            p2.AddRange(points2!);

            p1.AddRange(pose1r.points2!);
            p2.AddRange(pose1r.points1!);

            points1 = p1.ToArray();
            points2 = p2.ToArray();
        }

        internal Poser CrossCheck(Poser pose2, Mat mat1, Mat mat2, int windowSize = 5)
        {
            if (windowSize % 2 == 0)
                windowSize++; // Ensure odd window size for center pixel

            int halfWindow = windowSize / 2;

            List<Point2f> pts1 = new List<Point2f>();
            List<Point2f> pts2 = new List<Point2f>();
            List<Vec3b> clrs = new List<Vec3b>();

            for (int i = 0; i < points1!.Length; i++)
            {
                var p1 = points2![i];
                var p2 = pose2.points2![i];

                // Check if center points are valid
                if (mat1.InvalidPoint(p1) || mat2.InvalidPoint(p2))
                    continue;

                // Calculate window bounds for both images
                int x1_start = Math.Max(0, (int)p1.X - halfWindow);
                int y1_start = Math.Max(0, (int)p1.Y - halfWindow);
                int x1_end = Math.Min(mat1.Width - 1, (int)p1.X + halfWindow);
                int y1_end = Math.Min(mat1.Height - 1, (int)p1.Y + halfWindow);

                int x2_start = Math.Max(0, (int)p2.X - halfWindow);
                int y2_start = Math.Max(0, (int)p2.Y - halfWindow);
                int x2_end = Math.Min(mat2.Width - 1, (int)p2.X + halfWindow);
                int y2_end = Math.Min(mat2.Height - 1, (int)p2.Y + halfWindow);

                // Check if windows are the same size
                int width1 = x1_end - x1_start + 1;
                int height1 = y1_end - y1_start + 1;
                int width2 = x2_end - x2_start + 1;
                int height2 = y2_end - y2_start + 1;

                if (width1 != width2 || height1 != height2)
                    continue; // Skip if windows don't match (edge cases)

                // Compare windows
                double totalDiff = 0;
                int pixelCount = 0;

                for (int dy = 0; dy < height1; dy++)
                {
                    for (int dx = 0; dx < width1; dx++)
                    {
                        int img1_x = x1_start + dx;
                        int img1_y = y1_start + dy;
                        int img2_x = x2_start + dx;
                        int img2_y = y2_start + dy;

                        var co1 = mat1.At<Vec3b>(img1_y, img1_x);
                        var co2 = mat2.At<Vec3b>(img2_y, img2_x);

                        totalDiff += co2.SqDiff(co1);
                        pixelCount++;
                    }
                }

                // Calculate average difference per pixel
                double avgDiff = totalDiff / pixelCount;

                // Use threshold (adjustable based on your needs)
                if (avgDiff < 19)
                {
                    // Get center pixel color from mat1
                    var centerColor = mat1.At<Vec3b>((int)p1.Y, (int)p1.X);

                    pts1.Add(p1);
                    pts2.Add(p2);
                    clrs.Add(centerColor);
                }
            }

            var res = new Poser();
            res.points1 = pts1.ToArray();
            res.points2 = pts2.ToArray();
            res.colors = clrs.ToArray();

            return res;
        }

        internal void LoadPoints(string file)
        {
            var lns = File.ReadAllLines(file);
            var ttl = long.Parse(lns[0]);

            colors = new Vec3b[ttl];
            points1 = new Point2f[ttl];
            points2 = new Point2f[ttl];
            points3D = new Point3f[ttl];

            for (int i = 1; i < lns.Length; i++)
            {
                var line = lns[i];
                var sp = line.Split(' ');

                var b1 = byte.Parse(sp[0]);
                var b2 = byte.Parse(sp[1]);
                var b3 = byte.Parse(sp[2]);

                var v = new Vec3b(b1, b2, b3);
                colors[i - 1] = v;

                var x = float.Parse(sp[3]);
                var y = float.Parse(sp[4]);
                var pt = new Point2f(x, y);
                points1[i - 1] = pt;

                x = float.Parse(sp[5]);
                y = float.Parse(sp[6]);
                pt = new Point2f(x, y);
                points2[i - 1] = pt;

                x = float.Parse(sp[7]);
                y = float.Parse(sp[8]);
                var z = float.Parse(sp[9]);

                var pt3d = new Point3f(x, y, z);
                points3D[i - 1] = pt3d;
            }
        }


        internal void SavePoints(string file)
        {
            File.WriteAllText(file, colors!.Length.ToString() + "\r\n");

            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < colors!.Length; i++)
            {
                var co = colors![i];
                sb.Append(co.Item0!.ToString() + " ");
                sb.Append(co.Item1!.ToString() + " ");
                sb.Append(co.Item2!.ToString() + " ");

                sb.Append(points1![i].X + " ");
                sb.Append(points1![i].Y + " ");

                sb.Append(points2![i].X + " ");
                sb.Append(points2![i].Y + " ");

                sb.Append(points3D![i].X + " ");
                sb.Append(points3D![i].Y + " ");
                sb.Append(points3D![i].Z + "\r\n");
            }

            File.AppendAllText(file, sb.ToString());
        }

        internal Poser MergeWith(Poser pose2)
        {
            var affine = AffineTrans.GetAffine3d(base_points3D!, pose2.base_points3D!);
            var alignedPoints = AffineTrans.Transform(pose2.points3D!, affine);
            var alignedColors = pose2.colors!.ToArray();

            var res = new Poser();

            res.base_points3D = base_points3D.ToArray();
            res.base_points1 = base_points1.ToArray();
            res.base_points2 = base_points2.ToArray();
            res.base_points3D = base_points3D.ToArray();

            List<Point3f> allPoints3D = new List<Point3f>();
            List<Vec3b> allColors = new List<Vec3b>();

            allPoints3D.AddRange(points3D!);
            allColors.AddRange(colors);
            allPoints3D.AddRange(alignedPoints);
            allColors.AddRange(alignedColors);

            res.points3D = allPoints3D.ToArray();
            res.colors = allColors.ToArray();

            return res;
        }

        /// <summary>
        /// Moves point p2 closer to p1 by amount (when amount is positive)
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <param name="amount"></param>
        /// <returns>the translated point</returns>
        internal static Point3f TranslatePoint(Point3f p1,
            Point3f p2,
            float amount)
        {
            // Calculate the vector from p2 to p1
            float dx = p1.X - p2.X;
            float dy = p1.Y - p2.Y;
            float dz = p1.Z - p2.Z;

            // Calculate the distance between the points
            float distance = (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);

            // Handle case where points are coincident
            if (distance < 1e-6f)
            {
                return new Point3f(p2.X, p2.Y, p2.Z);
            }

            // Normalize the direction vector
            float unitX = dx / distance;
            float unitY = dy / distance;
            float unitZ = dz / distance;

            // Move p2 toward p1 by the specified amount
            float newX = p2.X + unitX * amount;
            float newY = p2.Y + unitY * amount;
            float newZ = p2.Z + unitZ * amount;

            return new Point3f(newX, newY, newZ);
        }

        internal void Scale()
        {
            var maxX = Math.Abs(points3D!.Max(s => Math.Abs(s.X)));
            var maxY = Math.Abs(points3D!.Max(s => Math.Abs(s.Y)));
            var maxZ = Math.Abs(points3D!.Max(s => Math.Abs(s.Z)));

            var maxV = Math.Max(maxX, Math.Max(maxY, maxZ));

            for (int i = 0; i < points3D!.Length; i++)
            {
                var p = points3D[i];
                p = new Point3f(p.X / maxV, p.Y / maxV, p.Z / maxV);
                points3D[i] = p;
            }
        }

        internal static void LogicalAND(params Poser[] posers)
        {
            if (posers == null || posers.Length == 0)
                return;

            int requiredCount = posers.Length;

            // Count how many posers contain each unique point
            var pointCount = new Dictionary<Point, int>();

            foreach (var poser in posers)
            {
                var uniquePoints = new HashSet<Point>(
                    poser.points1!.Select(p => p.ToPoint())
                );

                foreach (var point in uniquePoints)
                {
                    if (!pointCount.ContainsKey(point))
                        pointCount[point] = 0;
                    pointCount[point]++;
                }
            }

            // Get points that appear in ALL posers
            var commonPoints = pointCount
                .Where(kvp => kvp.Value >= requiredCount)
                .Select(kvp => kvp.Key)
                .ToList();

            // Filter each poser to keep only common points
            Parallel.For(0, posers.Length, poserIndex =>
            {
                var poser = posers[poserIndex];
                var pointsList = poser.points1.Select(p => p.ToPoint()).ToList();
                var keepIndices = new List<int>();

                foreach (var commonPoint in commonPoints)
                {
                    int idx = pointsList.IndexOf(commonPoint);
                    if (idx >= 0)
                        keepIndices.Add(idx);
                }

                // Update poser arrays with filtered points
                poser.base_points1 = keepIndices.Select(i => poser.points1[i]).ToArray();
                poser.base_points2 = keepIndices.Select(i => poser.points2[i]).ToArray();
                poser.base_points3D = keepIndices.Select(i => poser.points3D[i]).ToArray();
                poser.base_colors = keepIndices.Select(i => poser.colors[i]).ToArray();
            });
        }


        internal void CrossCheckTriplet(Poser pose1, Poser pose2)
        {
            // Build hash maps for O(1) lookups instead of O(n) List.FindIndex
            var pose2_p1_to_index = new Dictionary<Point, List<int>>();
            var this_p2_to_index = new Dictionary<Point, List<int>>();

            // Index pose2.points1 (images[0])
            for (int i = 0; i < pose2.points1.Length; i++)
            {
                var pt = pose2.points1[i].ToPoint();
                if (!pose2_p1_to_index.ContainsKey(pt))
                    pose2_p1_to_index[pt] = new List<int>();
                pose2_p1_to_index[pt].Add(i);
            }

            // Index this.points2 (images[2])
            for (int i = 0; i < this.points2.Length; i++)
            {
                var pt = this.points2[i].ToPoint();
                if (!this_p2_to_index.ContainsKey(pt))
                    this_p2_to_index[pt] = new List<int>();
                this_p2_to_index[pt].Add(i);
            }

            List<int> keep_pose1 = new List<int>();
            List<int> keep_pose2 = new List<int>();

            // Iterate through pose1 once
            for (int i = 0; i < pose1.points1.Length; i++)
            {
                var p1 = pose1.points1[i].ToPoint();  // images[0]
                var p2 = pose1.points2[i].ToPoint();  // images[1]

                // Does p1 exist in pose2.points1?
                if (pose2_p1_to_index.TryGetValue(p1, out var pose2_indices))
                {
                    foreach (var idx1 in pose2_indices)
                    {
                        var p3 = pose2.points2[idx1].ToPoint();  // images[2]

                        // Does p3 exist in this.points2?
                        if (this_p2_to_index.TryGetValue(p3, out var this_indices))
                        {
                            foreach (var idx2 in this_indices)
                            {
                                var p4 = this.points1[idx2].ToPoint();  // images[1]

                                // Check if the loop closes: p2 == p4
                                if (p2 == p4)
                                {
                                    keep_pose1.Add(i);
                                    keep_pose2.Add(idx1);
                                    goto NextPoint; // Found a valid match, move to next point
                                }
                            }
                        }
                    }
                }
            NextPoint:;
            }

            // Update pose1 arrays
            pose1.points1 = keep_pose1.Select(i => pose1.points1[i]).ToArray();
            pose1.points2 = keep_pose1.Select(i => pose1.points2[i]).ToArray();
            pose1.points3D = keep_pose1.Select(i => pose1.points3D[i]).ToArray();
            pose1.colors = keep_pose1.Select(i => pose1.colors[i]).ToArray();

            // Update pose2 arrays
            pose2.points1 = keep_pose2.Select(i => pose2.points1[i]).ToArray();
            pose2.points2 = keep_pose2.Select(i => pose2.points2[i]).ToArray();
            pose2.points3D = keep_pose2.Select(i => pose2.points3D[i]).ToArray();
            pose2.colors = keep_pose2.Select(i => pose2.colors[i]).ToArray();
        }
    }
}
