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
            var can1 = mtch.image1!.CannyExt(20, 40);
            var can2 = mtch.image2!.CannyExt(20, 40);

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

            //get squares that match
            for (int y = 0; y < can1.Height - sqsize; y += sqsize / 2)
            {
                for (int x1 = 0; x1 < can1.Width - sqsize; x1 += sqsize / 2)
                {
                    double maxp = 0;
                    int sx1 = 0, sx2 = 0;

                    //get the square
                    var rect1 = new Rect(x1, y, sqsize, sqsize);
                    var sq1 = can1[rect1];

                    //does it have enough information?
                    var sum1 = sq1.Sum();
                    var avg1 = sum1.Val0 / ttlpts / 255;

                    //find the square in the 2nd image
                    if (avg1 >= lowerb && avg1 <= upperb)
                    {
                        //we have info!

                        List<(double, int)> minfo = new List<(double, int)>();

                        //find sq1 inside the 2nd image
                        for (int x2 = x1 - wiggle; x2 < x1 + wiggle; x2++)
                        {
                            if (x2 < 0) continue;
                            if (x2 >= can2.Width - sqsize) continue;

                            //get the square
                            var rect2 = new Rect(x2, y, sqsize, sqsize);
                            var sq2 = can2[rect2];

                            var sum2 = sq2.Sum();
                            var avg2 = sum2.Val0 / ttlpts / 255;

                            if (avg2 >= lowerb && avg2 <= upperb)
                            {
                                //does it match?

                                sq1 = mtch.image1![rect1];
                                sq2 = mtch.image2![rect2];

                                var sq1r = sq1.GetChannel(0);
                                var sq2r = sq2.GetChannel(0);

                                var sq1g = sq1.GetChannel(1);
                                var sq2g = sq2.GetChannel(1);

                                var sq1b = sq1.GetChannel(2);
                                var sq2b = sq2.GetChannel(2);

                                //if so, save it...
                                var compR = sq1r.Compare(sq2r);
                                var compG = sq1g.Compare(sq2g);
                                var compB = sq1b.Compare(sq2b);

                                double comp = (compR + compG + compB) / 3d;

                                if (comp >= pvalue)
                                {
                                    if (comp > maxp)
                                    {
                                        maxp = comp;
                                        sx1 = x1;
                                        sx2 = x2;
                                    }
                                }
                            }
                        }
                    }

                    if (maxp > 0)
                    {
                        //add to the pMaps
                        for (int szX = 0; szX < sqsize; szX++)
                            for (int szY = 0; szY < sqsize; szY++)
                            {
                                if (maxp > pMaps[x1 + szX, y + szY].probability)
                                {
                                    pMaps[sx1 + szX, y + szY].probability = maxp;
                                    pMaps[sx1 + szX, y + szY].relative_pointX = sx2 + szX;
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
    }
}
