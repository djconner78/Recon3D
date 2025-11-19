using OpenCvSharp;
using Point = OpenCvSharp.Point;
using Rect = OpenCvSharp.Rect;
using Size = OpenCvSharp.Size;

namespace Recon3D
{
    internal static class MatExtensions
    {

        internal static Mat BalanceColors(this Mat image)
        {
            Mat src1 = image.Clone();
            var avgCo1 = new Vec3d();

            int ttl = src1.Width * src1.Height;

            for (int x = 0; x < src1.Width; x++)
            {
                for (int y = 0; y < src1.Height; y++)
                {
                    var pt = src1.At<Vec3b>(y, x);
                    avgCo1.Item0 += pt.Item0;
                    avgCo1.Item1 += pt.Item1;
                    avgCo1.Item2 += pt.Item2;
                }
            }

            avgCo1.Item0 /= ttl;
            avgCo1.Item1 /= ttl;
            avgCo1.Item2 /= ttl;

            BalanceCo(src1, avgCo1);
            return src1;
        }


        internal static double Compare(this Mat source, Mat image)
        {
            byte[] data1;
            source.GetArray<byte>(out data1);

            byte[] data2;
            image.GetArray<byte>(out data2);

            var data_d1 = data1.Select(s => (double)s).ToArray();
            var data_d2 = data2.Select(s => (double)s).ToArray();

            return data_d1.ComputeCoeff(data_d2);
        }




        static void BalanceCo(Mat src1, Vec3d avgCo1)
        {
            var dr = 127 - avgCo1.Item0;
            var dg = 127 - avgCo1.Item1;
            var db = 127 - avgCo1.Item2;

            Func<double, double, byte> checkv = (v, delta) =>
            {
                var x = v + delta;
                x = x < 0 ? 0 : x > 255 ? 255 : x;
                return (byte)x;
            };

            for (int x = 0; x < src1.Width; x++)
            {
                for (int y = 0; y < src1.Height; y++)
                {
                    var co = src1.GetColor(x, y);
                    co.Item0 = checkv(co.Item0, dr);
                    co.Item1 = checkv(co.Item1, dg);
                    co.Item2 = checkv(co.Item2, db);

                    src1.Set<Vec3b>(y, x, co);
                }
            }
        }


        internal static Mat ToBGRGray(this Mat src)
        {
            var tmp = src.Type() == MatType.CV_8UC1 ? src.Clone() : src.CvtColor(ColorConversionCodes.BGR2GRAY);
            var res = tmp.CvtColor(ColorConversionCodes.GRAY2BGR);
            tmp.Dispose();
            return res;
        }



        internal static Vec3b GetColor(this Mat src, int x, int y)
        {
            return src.At<Vec3b>(y, x);
        }


        internal static Vec3b GetColor(this Mat src, Point pt)
        {
            return src.At<Vec3b>(pt.Y, pt.X);
        }

        internal static Vec3b GetColor(this Mat src, Point2f pt2f)
        {
            var pt = (Point)pt2f;
            return src.At<Vec3b>(pt.Y, pt.X);
        }

        internal static bool InvalidPoint(this Mat img, Point point)
        {
            return point.X < 0 || point.X >= img.Width || point.Y < 0 || point.Y >= img.Height;
        }

        internal static bool InvalidPoint(this Mat img, Point2f point)
        {
            return InvalidPoint(img, (Point)point);
        }

        internal static Mat BitwiseNot(this Mat img)
        {
            var res = new Mat();
            Cv2.BitwiseNot(img, res);
            return res;
        }



        internal static KeyPoint[] ToKeyPoints(this Point2f[] array)
        {
            return array.Select(s => new KeyPoint(s, 1)).ToArray();
        }


        internal static Point3d ToPoint3d(this Point3f point)
        {
            return new Point3d(point.X, point.Y, point.Z);
        }


        internal static float DistanceTo(this Point3f src, Point3f point)
        {
            var dx = point.X - src.X;
            var dy = point.Y - src.Y;
            var dz = point.Z - src.Z;

            return (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }


        internal static Mat ApplyMask(this Mat img, Mat mask)
        {
            var res = img.Clone();
            var black = new Vec3b(0, 0, 0);

            for (int i = 0; i < img.Width; i++)
            {
                for (int j = 0; j < img.Height; j++)
                {
                    var t = mask.GetColor(i, j);
                    if (t == black)
                    {
                        res.Set<Vec3b>(j, i, t);
                    }
                }
            }

            return res;
        }


        internal static double GetColorDifference(Vec3b p1, Vec3b p2)
        {
            var sum = (double)((p1.Item0 - p2.Item0)
                * (p1.Item0 - p2.Item0)
                + (p1.Item1 - p2.Item1)
                * (p1.Item1 - p2.Item1)
                + (p1.Item2 - p2.Item2)
                * (p1.Item2 - p2.Item2));

            return sum;
        }


        internal static double SqDiff(this Vec3b co1, Vec3b co2)
        {
            return GetColorDifference(co1, co2);
        }


        internal static Double PearsonCorrelation(this Vec3b[] co1, Vec3b[] co2, bool decorrelate = false)
        {
            var co1_i0 = co1.Select(s => (double)s.Item0).ToArray();
            var co2_i0 = co2.Select(s => (double)s.Item0).ToArray();

            if (decorrelate)
            {
                co1_i0 = Detrend(co1_i0);
                co2_i0 = Detrend(co2_i0);
            }

            var p0 = PearsonCorrelation(co1_i0, co2_i0);

            var co1_i1 = co1.Select(s => (double)s.Item1).ToArray();
            var co2_i1 = co2.Select(s => (double)s.Item1).ToArray();

            if (decorrelate)
            {
                co1_i1 = Detrend(co1_i1);
                co2_i1 = Detrend(co2_i1);
            }

            var p1 = PearsonCorrelation(co1_i1, co2_i1);

            var co1_i2 = co1.Select(s => (double)s.Item2).ToArray();
            var co2_i2 = co2.Select(s => (double)s.Item2).ToArray();

            if (decorrelate)
            {
                co1_i2 = Detrend(co1_i2);
                co2_i2 = Detrend(co2_i2);
            }

            var p2 = PearsonCorrelation(co1_i2, co2_i2);

            return (p0 + p1 + p2) / 3d;
        }

        private static double[] Detrend(double[] v)
        {
            var d1 = v.Select((d, idx) => (idx == 0 ? 0 : d - v[idx - 1]))
                .Skip(1)
                .ToArray();

            return d1;
        }

        public static double ComputeCoeff(this double[] values1, double[] values2)
        {
            if (values1.Length != values2.Length)
                throw new ArgumentException("values must be the same length");

            var avg1 = values1.Average();
            var avg2 = values2.Average();

            var sum1 = values1.Zip(values2, (x1, y1) => (x1 - avg1) * (y1 - avg2)).Sum();

            var sumSqr1 = values1.Sum(x => Math.Pow((x - avg1), 2.0));
            var sumSqr2 = values2.Sum(y => Math.Pow((y - avg2), 2.0));

            var result = sum1 / Math.Sqrt(sumSqr1 * sumSqr2);

            return result;
        }

        internal static Double PearsonCorrelation(Double[] Xs, Double[] Ys)
        {
            Double sumX = 0;
            Double sumX2 = 0;
            Double sumY = 0;
            Double sumY2 = 0;
            Double sumXY = 0;

            int n = Xs.Length < Ys.Length ? Xs.Length : Ys.Length;

            for (int i = 0; i < n; ++i)
            {
                Double x = Xs[i];
                Double y = Ys[i];

                sumX += x;
                sumX2 += x * x;
                sumY += y;
                sumY2 += y * y;
                sumXY += x * y;
            }

            Double stdX = Math.Sqrt(sumX2 / n - sumX * sumX / n / n);
            Double stdY = Math.Sqrt(sumY2 / n - sumY * sumY / n / n);
            Double covariance = (sumXY / n - sumX * sumY / n / n);

            return covariance / stdX / stdY;
        }


        internal static Point[] GetPointsRounded(this KeyPoint[] keypoints)
        {
            var pts = keypoints.Select(s => new Point((int)Math.Round(s.Pt.X),
                (int)Math.Round(s.Pt.Y))).ToArray();
            return pts;
        }


        internal static Mat GetChannel(this Mat src, int channel)
        {
            var sp = src.Split();
            return sp[channel];
        }


        internal static void NormalizeWith(this Mat img1, Mat img2, Point[] points1, Point[] points2)
        {
            var co1 = points1.Select(s => img1.GetColor(s)).ToArray();
            var co2 = points2.Select(s => img2.GetColor(s)).ToArray();

            var avgR1 = co1.Select((s, idx) => (double)co2[idx].Item0 - s.Item0).Average();
            var avgG1 = co1.Select((s, idx) => (double)co2[idx].Item1 - s.Item1).Average();
            var avgB1 = co1.Select((s, idx) => (double)co2[idx].Item2 - s.Item2).Average();

            for (int y = 0; y < img1.Height; y++)
            {
                for (int x = 0; x < img1.Width; x++)
                {
                    var co = img1.GetColor(x, y);
                    var nco = AddAmount(co, avgR1, avgG1, avgB1);
                    img1.Set<Vec3b>(y, x, nco);
                }
            }
        }


        internal static Point2f[] ToPoints2f(this KeyPoint[] kp)
        {
            return kp.Select(s => s.Pt).ToArray();
        }

        private static Vec3b AddAmount(Vec3b co, double avgR1, double avgG1, double avgB1)
        {
            var c1 = co.Item0 + avgR1;
            var c2 = co.Item1 + avgG1;
            var c3 = co.Item2 + avgB1;

            c1 = c1 < 0 ? 0 : c1 > 255 ? 255 : c1;
            c2 = c2 < 0 ? 0 : c2 > 255 ? 255 : c2;
            c3 = c3 < 0 ? 0 : c3 > 255 ? 255 : c3;

            return new Vec3b((byte)c1, (byte)c2, (byte)c3);
        }

        internal static Vec3b Avg(this Vec3b[] stack)
        {
            var lastco_0 = (byte)stack.Average(a => a.Item0);
            var lastco_1 = (byte)stack.Average(a => a.Item1);
            var lastco_2 = (byte)stack.Average(a => a.Item2);

            return new Vec3b(lastco_0, lastco_1, lastco_2);
        }

        internal static Mat CannyExt(this Mat img, double t1, double t2)
        {
            var c1 = img.GetChannel(0);
            var c2 = img.GetChannel(1);
            var c3 = img.GetChannel(2);

            var can1 = c1.Canny(t1, t2);
            var can2 = c2.Canny(t1, t2);
            var can3 = c3.Canny(t1, t2);

            return can1.BitwiseOr(can2).ToMat().BitwiseOr(can3).ToMat();
        }

    }

    internal static class Vec3bExtensions
    {
        internal static bool IsEqual(this Vec3b co, Vec3b co2)
        {
            if (co.GetHashCode() != co2.GetHashCode())
                return false;
            return co.Item0 == co2.Item0 && co.Item1 == co2.Item1 && co.Item2 == co2.Item2;
        }
    }
}
