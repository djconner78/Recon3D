using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Recon3D
{
    internal class Triangulation
    {
        internal static void Triangulate(Mat cm, 
            Point2f[] src1, 
            Point2f[] src2,
            out Mat R,
            out Mat t,
            out Point3f[] ProjectedPoints)
        {
            var pts1 = Mat.FromArray<Point2f>(src1);
            var pts2 = Mat.FromArray<Point2f>(src2);

            //find essential matrix
            var ess = Cv2.FindEssentialMat(pts1, pts2, cm, EssentialMatMethod.Ransac);

            //get the camera position
            R = new Mat();
            t = new Mat();
            var cnt = Cv2.RecoverPose(ess, pts1, pts2, cm, R, t);

            //triangulate the points
            Mat pts3 = new Mat();
            Mat projMat1 = cm * Mat.Eye(3, 4, MatType.CV_64F);

            Mat extrinsicMat = new Mat(3, 4, MatType.CV_64F);
            using (Mat R_roi = new Mat(extrinsicMat, new Rect(0, 0, 3, 3)))
            {
                R.CopyTo(R_roi);
            }
            using (Mat t_roi = new Mat(extrinsicMat, new Rect(3, 0, 1, 3)))
            {
                t.CopyTo(t_roi);
            }

            Mat projMat2 = cm * extrinsicMat;

            Cv2.TriangulatePoints(projMat1,
                projMat2, pts1, pts2, pts3);

            Mat point_3d_space = new Mat();

            var tr = pts3.T();

            Cv2.ConvertPointsFromHomogeneous(tr,
                point_3d_space);

            point_3d_space.GetArray<Point3f>(out ProjectedPoints);
        }


        internal enum CamNumber
        {
            One = 1,
            Two = 2
        }

        internal static Point2f[] ReprojectPoints(Point3f[] points,
            Mat K,
            Mat R,
            Mat t,
            Mat distCoeffs,
            CamNumber camNumber)
        {
            // Convert 3x3 rotation matrix R to a 3x1 rotation vector rvec
            Mat rvec = new Mat();
            Cv2.Rodrigues(R, rvec);

            // Prepare the output matrix for the re-projected 2D points
            Mat reprojectedPoints = new Mat();

            // Create the array of points
            var cart = Mat.FromArray<Point3f>(points);

            if (camNumber == CamNumber.One)
            {
                rvec = new Mat(3, 1, MatType.CV_64F);
                rvec.SetTo(new Scalar(0));

                t = new Mat(3, 1, MatType.CV_64F);
                t.SetTo(new Scalar(0));
            }

            // Now, call the ProjectPoints function
            Cv2.ProjectPoints(cart.T(),
                rvec,
                t,
                K,
                distCoeffs,
                reprojectedPoints);

            var pts = new Point2f[0];
            reprojectedPoints.GetArray<Point2f>(out pts);

            return pts;
        }

        internal static Mat GetDistCoeff()
        {
            Mat distCoeffs = new Mat(5, 1, MatType.CV_64F);

            // Set all the values in the matrix to zero.
            distCoeffs.SetTo(new Scalar(0));

            return distCoeffs;
        }
    }
}
