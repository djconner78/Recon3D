using OpenCvSharp;
using System.Xml.Linq;

namespace Recon3D
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //create the output folder
            CreateOutputFolder();

            //get the image file names
            var ilist = new DirectoryInfo("Images").GetFiles()
                .Select(s => s.FullName).Take(4);

            //load the images and resize them
            var images = ilist.Select(s => new Mat(s, ImreadModes.Color)).ToArray();
            images = images.Select(s =>
            {
                //simple color balance and gaussian blur 
                var r = s.BalanceColors().GaussianBlur(new Size(), 1.5, 1.5);
                var dst = new Mat();

                //resize the images to something that can be
                //processed faster
                Cv2.Resize(r, dst, new Size(r.Width / 4, r.Height / 4));
                return dst;
            }).ToArray();


            //images are (1008, 756) pix
            
            float HFOV = 80f;  //this needs to be close to the actual horizontal field of view (HFOV)

            //create the 1st pose, dense cloud
            var pose1 = Poser.Create(images[0], images[1], HFOV);

            pose1.ExportOBJ("output/pose1.obj");

            //reproject the points back into 2d space
            pose1.ReprojectPoints("output/cam1.png", "output/cam2.png", images[0].Size());

            var pose2 = Poser.Create(images[0], images[2], HFOV);
            var pose3 = Poser.Create(images[0], images[3], HFOV);

            //perform a logical AND of all of the models
            //as this will remove noise at the expense
            //of some signal loss
            pose1.KeepSamePoints(pose2);
            pose1.KeepSamePoints(pose3);
            pose2.KeepSamePoints(pose3);

            pose1.ExportOBJ("output/pose_Logical_AND.obj");

            //reproject the points back into 2D space
            pose1.ReprojectPoints("output/Logical_AND_cam1.png", "output/Logical_AND_cam2.png", images[0].Size());

        }


        internal static void CreateOutputFolder()
        {
            var dinfo = new DirectoryInfo("output");
            if (!dinfo.Exists)
            {
                dinfo.Create();
            }
        }
    }
}
