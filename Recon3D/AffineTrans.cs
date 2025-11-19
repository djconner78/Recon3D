using OpenCvSharp;
using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace Recon3D
{
    internal class AffineTrans
    {
        internal static Point3f[] Transform(Point3f[] points2, Mat transform)
        {
            Point3f[] res = new Point3f[0];
            var src = Mat.FromArray(points2);
            var ndst = new Mat();

            Cv2.Transform(src, ndst, transform);

            _ = ndst.GetArray<Point3f>(out res);
            return res;
        }

        internal static Mat GetAffine3d(Point3f[] target, Point3f[] source)
        {
            Mat inliers = new Mat();
            Mat ouv = new Mat();
            Mat src = Mat.FromArray<Point3f>(source);
            Mat dst = Mat.FromArray<Point3f>(target);

            var ttl = Cv2.EstimateAffine3D(src, dst,
                ouv, inliers, 1, confidence: 0.999995d);

            return ouv;
        }


    }
}
