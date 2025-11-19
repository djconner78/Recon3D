using OpenCvSharp;
using OpenCvSharp.Features2D;
using System.Text;
using Color = System.Drawing.Color;
using Point = OpenCvSharp.Point;
using Rect = OpenCvSharp.Rect;
using Size = OpenCvSharp.Size;

namespace Recon3D
{
    internal class Matched
    {
        internal Mat F = new Mat();
        internal Mat H1 = new Mat();
        internal Mat H2 = new Mat();


        internal Point3f[] ProjectedPoints = new Point3f[0];
        internal Vec3b[] SourceColors = new Vec3b[0];

        internal Size imageSize;

        internal Mat? image1, image2;
        internal KeyPoint[] keypoints1, keypoints2;
        internal Mat? mask1;
        internal Mat? mask2;

        internal static Matched FindMatches(Mat image1,
            Mat image2,
            float distRatio = .75f,
            Mat? mask1 = null,
            Mat? mask2 = null,
            KeyPoint[]? existingPoints1 = null,
            KeyPoint[]? existingPoints2 = null)
        {
            var sift = SIFT.Create(2000000);
            var matched = new Matched(image1.Size());

            // Detect the keypoints and generate their descriptors using SIFT

            var descriptors1 = new Mat();   //<float>
            var descriptors2 = new Mat();   //<float>

            if (existingPoints1 != null)
            {
                sift.Compute(image1, ref existingPoints1, descriptors1);
                matched.keypoints1 = existingPoints1.ToArray();
            }
            else
            {
                sift.DetectAndCompute(image1,
                    mask1!, out matched.keypoints1, descriptors1);
            }

            if (existingPoints2 != null)
            {
                sift.Compute(image2, ref existingPoints2, descriptors2);
                matched.keypoints2 = existingPoints2.ToArray();
            }
            else
            {
                sift.DetectAndCompute(image2,
                    mask2!, out matched.keypoints2, descriptors2);
            }

            if (!matched.keypoints1.Any() || !matched.keypoints2.Any())
            {
                sift.Dispose();
                descriptors1.Dispose();
                descriptors2.Dispose();
                matched = new Matched(image1.Size());
                matched.image1 = image1;
                matched.image2 = image2;
                matched.keypoints1 = new KeyPoint[0];
                matched.keypoints2 = new KeyPoint[0];
                return matched;
            }

            var bfMatcher = new FlannBasedMatcher();

            DMatch[][] bfMatches;
            try
            {
                bfMatches = bfMatcher.KnnMatch(descriptors1,
                    descriptors2, 2);
            }
            catch
            {
                bfMatches = new DMatch[0][];
                matched.keypoints1 = new KeyPoint[0];
                matched.keypoints2 = new KeyPoint[0];
            }

            matched.image1 = image1;
            matched.image2 = image2;


            List<KeyPoint> keep1 = new List<KeyPoint>();
            List<KeyPoint> keep2 = new List<KeyPoint>();

            foreach (var match in bfMatches)
            {
                if (match[0].Distance < distRatio * match[1].Distance)
                {
                    var ky1 = matched.keypoints1[match[0].QueryIdx];
                    var ky2 = matched.keypoints2[match[0].TrainIdx];

                    keep1.Add(ky1);
                    keep2.Add(ky2);
                }
            }

            matched.keypoints1 = keep1.ToArray();
            matched.keypoints2 = keep2.ToArray();

            bfMatcher.Dispose();
            sift.Dispose();
            descriptors1.Dispose();
            descriptors2.Dispose();

            return matched;
        }

        internal Matched(Size imageSize)
        {
            keypoints1 = new KeyPoint[0];
            keypoints2 = new KeyPoint[0];
            this.imageSize = imageSize;
        }


        internal void CalculateFundamentalAndFilter(bool correctPoints = false)
        {
            var mask = new Mat();

            List<Point2f> kp1 = keypoints1.Select(s => s.Pt).ToList();
            List<Point2f> kp2 = keypoints2.Select(s => s.Pt).ToList();

            F = Cv2.FindFundamentalMat(kp1,
                    kp2,
                    FundamentalMatMethods.Ransac,
                    2, 0.99d, mask);

            if (F.IsContinuous())
            {
                if (F.Width == 3 && F.Height == 3)
                {
                    if (correctPoints)
                    {
                        CorrectPoints();
                    }
                    keypoints1 = keypoints1.Where((v, idx) => mask.At<byte>(0, idx) == 1).ToArray();
                    keypoints2 = keypoints2.Where((v, idx) => mask.At<byte>(0, idx) == 1).ToArray();
                }
            }
        }


        internal void CalculateH1H2()
        {
            List<Point2f> kp1 = keypoints1.Select(s => s.Pt).ToList();
            List<Point2f> kp2 = keypoints2.Select(s => s.Pt).ToList();

            var src1 = Mat.FromArray<Point2f>(kp1);
            var src2 = Mat.FromArray<Point2f>(kp2);

            var success = Cv2.StereoRectifyUncalibrated(src1,
                src2, F, imageSize, H1, H2);

            if (!success)
                throw new Exception("could not calibrate");
        }

        internal void SetSourceColors1()
        {
            var kp1 = keypoints1.Select(s => s.Pt).ToList();

            var cos = new List<Vec3b>();

            for (int i = 0; i < kp1.Count; i++)
            {
                var pt = kp1[i].ToPoint();

                var co = new Vec3b();
                if (!image1!.InvalidPoint(pt))
                {
                    co = image1!.GetColor(pt);
                }

                cos.Add(co);
            }

            SourceColors = cos.ToArray();
        }

        internal Point2f[] GetPoints1()
        {
            return keypoints1.Select(s => s.Pt).ToArray();
        }

        internal Point2f[] GetPoints2()
        {
            return keypoints2.Select(s => s.Pt).ToArray();
        }


        internal void FilterSimilarYValues(float tolerance)
        {
            List<Point2f> kp1 = keypoints1.Select(s => s.Pt).ToList();
            List<Point2f> kp2 = keypoints2.Select(s => s.Pt).ToList();

            var k1 = kp1.Select((s, idx) => new { index = idx, ydiff = Math.Abs(s.Y - kp2[idx].Y) }).ToList();
            var good = k1.Where(w => w.ydiff <= tolerance).Select(s => s.index).ToList();

            keypoints1 = good.Select(s => keypoints1[s]).ToArray();
            keypoints2 = good.Select(s => keypoints2[s]).ToArray();
        }

        internal void FilterSimilarSizes(double tolerance)
        {
            List<Point2f> kp1 = keypoints1.Select(s => s.Pt).ToList();
            List<Point2f> kp2 = keypoints2.Select(s => s.Pt).ToList();

            var sz = keypoints1.Select(s => s.Size).Select((s, idx) => new { index = idx, ratio = s / keypoints2[idx].Size }).ToList();
            var inv = 1 / tolerance;

            var good = sz.Where(w => w.ratio >= tolerance && w.ratio <= inv).ToList();

            keypoints1 = good.Select(s => keypoints1[s.index]).ToArray();
            keypoints2 = good.Select(s => keypoints2[s.index]).ToArray();
        }

        internal Mat GetCameraMatrix(float HFOV)
        {
            float width = imageSize.Width;
            float height = imageSize.Height;

            float cx = (width - 1) / 2f;
            float cy = (height - 1) / 2f;

            float VFOV = HFOV / cx * cy;

            // abs(Float(dim.width) / (2 * tan(HFOV / 180 * Float(M_PI) / 2)));
            float fx = (float)Math.Abs(width / (2 * Math.Tan(HFOV / 180 * Math.PI / 2)));
            float fy = (float)Math.Abs(height / (2 * Math.Tan(VFOV / 180 * Math.PI / 2)));

            //[fx,0,cx]
            //[0,fy,cy]
            //[0,0,1]
            var mat = new Mat(new Size(3, 3), MatType.CV_64F);

            mat.Set<double>(0, 0, fx);
            mat.Set<double>(0, 1, 0);
            mat.Set<double>(0, 2, cx);

            mat.Set<double>(1, 0, 0);
            mat.Set<double>(1, 1, fy);
            mat.Set<double>(1, 2, cy);

            mat.Set<double>(2, 0, 0);
            mat.Set<double>(2, 1, 0);
            mat.Set<double>(2, 2, 1);

            return mat;
        }

        internal void ExportPoints(string target)
        {
            var matched = this;

            using (var strm = new FileStream(target, FileMode.Create))
            {
                for (int i = 0; i < matched.SourceColors.Length; i++)
                {
                    var co = matched.SourceColors[i];
                    var t = matched.ProjectedPoints[i];

                    var mx = Math.Max(Math.Max(Math.Abs(t.X), Math.Abs(t.Y)), Math.Abs(t.Z));

                    if (mx < 100000)
                    {
                        var ln = $"{t.X} {t.Y * -1} {t.Z * -1} {co.Item2} {co.Item1} {co.Item0}\r\n";
                        //System.IO.File.AppendAllText("test.xyzc", ln);
                        var bts = System.Text.UTF8Encoding.UTF8.GetBytes(ln);
                        strm.Write(bts, 0, bts.Length);
                    }
                }
                strm.Flush();
                strm.Close();
            }
        }


        /// <summary>
        /// Warps the transform
        /// </summary>
        /// <returns>Warp with mask</returns>
        internal (Mat, Mat) WarpTransform1H1()
        {
            var img3 = new Mat();
            Cv2.WarpPerspective(image1!, img3, H1, image1!.Size(), InterpolationFlags.Linear, BorderTypes.Constant);

            var rect = new Rect(0, 0, image1.Width, image1.Height);
            var mask = image1.EmptyClone();
            mask.Rectangle(rect, new Scalar(255, 255, 255), -1);

            var wmask = new Mat();
            Cv2.WarpPerspective(mask, wmask, H1, image1.Size(), InterpolationFlags.Linear, BorderTypes.Constant);

            return (img3, wmask);
        }

        internal (Mat, Mat) WarpTransform2H2()
        {
            var img3 = new Mat();
            Cv2.WarpPerspective(image2!, img3, H2, image2!.Size(), InterpolationFlags.Linear, BorderTypes.Constant);

            var rect = new Rect(0, 0, image2.Width, image2.Height);
            var mask = image2.EmptyClone();
            mask.Rectangle(rect, new Scalar(255, 255, 255), -1);

            var wmask = new Mat();
            Cv2.WarpPerspective(mask, wmask, H2, image2.Size(), InterpolationFlags.Linear, BorderTypes.Constant);

            return (img3, wmask);
        }

        internal void WarpTransforms()
        {
            var r1 = WarpTransform1H1();
            image1 = r1.Item1;

            var r2 = WarpTransform2H2();
            image2 = r2.Item1;
        }

        internal void UnWarpPointsH1H2()
        {
            if (!keypoints1.Any())
                return;

            Mat INV1 = H1.Inv();

            var npts1 = Cv2.PerspectiveTransform(keypoints1.Select(s => new Point2d(s.Pt.X, s.Pt.Y)).ToArray(),
                INV1);
            INV1.Dispose();

            keypoints1 = npts1.Select(s => new KeyPoint(new Point2f((float)s.X, (float)s.Y), 1)).ToArray();

            Mat INV2 = H2.Inv();
            var npts2 = Cv2.PerspectiveTransform(keypoints2.Select(s => new Point2d(s.Pt.X, s.Pt.Y)).ToArray(),
                INV2);
            INV2.Dispose();

            keypoints2 = npts2.Select(s => new KeyPoint(new Point2f((float)s.X, (float)s.Y), 1)).ToArray();
        }

        internal void WarpPointsH1H2()
        {
            var npts1 = Cv2.PerspectiveTransform(
                keypoints1.Select(s => s.Pt).ToArray(),
                H1);

            keypoints1 = npts1.Select(s => new KeyPoint(s, 1)).ToArray();

            var npts2 = Cv2.PerspectiveTransform(
                keypoints2.Select(s => s.Pt).ToArray(),
                H2);

            keypoints2 = npts2.Select(s => new KeyPoint(s, 1)).ToArray();
        }



        internal void CorrectPoints()
        {
            var points1 = Mat.FromArray<Point2f>(keypoints1.Select(s => s.Pt).ToArray());
            var points2 = Mat.FromArray<Point2f>(keypoints2.Select(s => s.Pt).ToArray());
            var nPoints1 = new Mat();
            var nPoints2 = new Mat();

            points1 = points1.T();
            points2 = points2.T();

            Cv2.CorrectMatches(F, points1, points2,
                nPoints1, nPoints2);

            //nPoints1 = nPoints1.T();
            //nPoints2 = nPoints2.T();

            Point2f[] pts1 = new Point2f[0];
            Point2f[] pts2 = new Point2f[0];

            nPoints1.GetArray<Point2f>(out pts1);
            nPoints2.GetArray<Point2f>(out pts2);

            keypoints1 = pts1.Select(p => new KeyPoint(p, 1)).ToArray();
            keypoints2 = pts2.Select(p => new KeyPoint(p, 1)).ToArray();
        }


        internal Mat WarpTransformH1Mask(int margin = 0)
        {
            var img3 = new Mat();
            var i = Mat.Zeros(image1!.Size(), image1.Type()).ToMat();
            var rct = new Rect(margin, margin, i.Width - 2 * margin, i.Height - 2 * margin);
            i.Rectangle(rct, new Scalar(255, 255, 255), -1);
            Cv2.WarpPerspective(i, img3, H1, i.Size(), InterpolationFlags.Nearest, BorderTypes.Constant);
            return img3.CvtColor(ColorConversionCodes.BGR2GRAY);
        }


        internal Mat WarpTransformH2Mask(int margin = 0)
        {
            var img3 = new Mat();
            var i = Mat.Zeros(image2!.Size(), image2.Type()).ToMat();
            var rct = new Rect(margin, margin, image2.Width - 2 * margin, image2.Height - 2 * margin);
            i.Rectangle(rct, new Scalar(255, 255, 255), -1);
            Cv2.WarpPerspective(i, img3, H2, image2.Size(), InterpolationFlags.Nearest, BorderTypes.Constant);
            return img3.CvtColor(ColorConversionCodes.BGR2GRAY);
        }


        internal void WarpTransformsWithMasks(int margin = 0)
        {
            WarpTransforms();
            mask1 = WarpTransformH1Mask(margin);
            mask2 = WarpTransformH2Mask(margin);
        }


        internal void FilterSharedPoints_Key1(KeyPoint[] keypoints)
        {
            List<int> keep = new List<int>();
            var hset = keypoints1.Select(pt => pt.Pt).ToList();

            foreach (var pt in keypoints)
            {
                var idx = hset.IndexOf(pt.Pt);
                if (idx > -1)
                {
                    keep.Add(idx);
                }
            }

            keypoints1 = keep.Select(s => keypoints1[s]).ToArray();
            keypoints2 = keep.Select(s => keypoints2[s]).ToArray();

            if (ProjectedPoints.Any())
                ProjectedPoints = keep.Select(s => ProjectedPoints[s]).ToArray();

            if (SourceColors.Any())
                SourceColors = keep.Select(s => SourceColors[s]).ToArray();
        }

        internal void FilterSharedPoints_Key2(KeyPoint[] keypoints)
        {
            List<int> keep = new List<int>();
            var hset = keypoints2.Select(pt => pt.Pt).ToList();

            foreach (var pt in keypoints)
            {
                var idx = hset.IndexOf(pt.Pt);
                if (idx > -1)
                {
                    keep.Add(idx);
                }
            }

            keypoints1 = keep.Select(s => keypoints1[s]).ToArray();
            keypoints2 = keep.Select(s => keypoints2[s]).ToArray();

            if (ProjectedPoints.Any())
                ProjectedPoints = keep.Select(s => ProjectedPoints[s]).ToArray();
        }


        internal void FilterSharedAbsPoints(Matched m2)
        {
            List<int> keep1 = new List<int>();
            List<int> keep2 = new List<int>();

            var hset = keypoints1.Select(s => s.Pt.ToPoint()).ToList();

            //for (int i = 0; i < m2.keypoints1.Length; i++)
            Parallel.For(0, m2.keypoints1.Length, i =>
            {
                var pt = m2.keypoints1[i];
                var idx = hset.IndexOf(pt.Pt.ToPoint());
                if (idx > -1)
                {
                    lock (keep1)
                    {
                        keep1.Add(idx);
                        keep2.Add(i);
                    }
                }
            });

            keypoints1 = keep1.Select(s => keypoints1[s]).ToArray();
            keypoints2 = keep1.Select(s => keypoints2[s]).ToArray();
            SourceColors = keep1.Select(s => SourceColors[s]).ToArray();
            ProjectedPoints = keep1.Select(s => ProjectedPoints[s]).ToArray();

            m2.keypoints1 = keep2.Select(s => m2.keypoints1[s]).ToArray();
            m2.keypoints2 = keep2.Select(s => m2.keypoints2[s]).ToArray();
            m2.SourceColors = keep2.Select(s => m2.SourceColors[s]).ToArray();
            m2.ProjectedPoints = keep2.Select(s => m2.ProjectedPoints[s]).ToArray();
        }


        internal void Dispose()
        {
            F?.Dispose();
            H1?.Dispose();
            H2?.Dispose();

            image1?.Dispose();
            image2?.Dispose();
            mask1?.Dispose();
            mask2?.Dispose();
        }


        internal void ExportOBJ(string fileName)
        {

            StringBuilder res = new StringBuilder();

            for (int i = 0; i < ProjectedPoints.Length; i++)
            {
                var pp = ProjectedPoints[i];
                var cp = SourceColors[i];
                var ln = $"v {pp.X} {pp.Y} {pp.Z} {cp.Item2 / 255f} {cp.Item1 / 255f} {cp.Item0 / 255f}\r\n";
                res.Append(ln);
            }

            File.WriteAllText(fileName, res.ToString());
        }



        internal void ExportPLY(string fileName)
        {
            string pcdheader = $@"ply
format ascii 1.0
comment Point cloud with RGB colors
element vertex {ProjectedPoints.Length}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
";
            StringBuilder res = new StringBuilder();
            res.Append(pcdheader);

            for (int i = 0; i < ProjectedPoints.Length; i++)
            {
                var pp = ProjectedPoints[i];
                var cp = SourceColors[i];
                var ln = $"{pp.X} {pp.Y} {pp.Z} {cp.Item2} {cp.Item1} {cp.Item0}\r\n";
                res.Append(ln);
            }

            File.WriteAllText(fileName, res.ToString());
        }




        internal void ExportPCD(string fileName)
        {
            string pcdheader = $@"# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {ProjectedPoints.Length}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {ProjectedPoints.Length}
DATA ascii
";
            StringBuilder res = new StringBuilder();
            res.Append(pcdheader);

            for (int i = 0; i < ProjectedPoints.Length; i++)
            {
                var pp = ProjectedPoints[i];
                var cp = SourceColors[i];

                var uu = (float)RgbToFloat(cp.Item2, cp.Item1, cp.Item0);

                var ln = $"{pp.X} {pp.Y} {pp.Z} {uu}\r\n";
                //pcdheader += ln;
                res.Append(ln);
            }

            File.WriteAllText(fileName, res.ToString());
        }


        public static float RgbToFloat(byte r, byte g, byte b)
        {
            uint rgbInt = ((uint)r << 16) | ((uint)g << 8) | (uint)b;
            return BitConverter.ToSingle(BitConverter.GetBytes(rgbInt), 0);
        }
    }
}
