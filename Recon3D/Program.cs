using OpenCvSharp;
using System;
using System.Text.RegularExpressions;
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
                .Select(s => s.FullName).Take(6);

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

            //create the other poses
            var pose2 = Poser.Create(images[0], images[2], HFOV);
            var pose2r = Poser.Create(images[1], images[2], HFOV);

            var pose3 = Poser.Create(images[0], images[3], HFOV);
            var pose3r = Poser.Create(images[2], images[3], HFOV);

            var pose4 = Poser.Create(images[0], images[4], HFOV);
            var pose4r = Poser.Create(images[3], images[4], HFOV);

            var pose5 = Poser.Create(images[0], images[5], HFOV);
            var pose5r = Poser.Create(images[4], images[5], HFOV);

            //cross check the triplets to remove outliers
            //and reduce noise
            pose2r.CrossCheckTriplet(pose1, pose2);
            pose3r.CrossCheckTriplet(pose2, pose3);
            pose4r.CrossCheckTriplet(pose3, pose4);
            pose5r.CrossCheckTriplet(pose4, pose5);

            pose1.ExportOBJ("output/pose1_after_triplet.obj");

            //merge all models together

            //perform a logical AND of all of the models
            //as this will remove noise at the expense
            //of some signal loss

            Poser.LogicalAND(pose1, pose2, pose3, pose4, pose5);
            pose1.ExportOBJ("output/pose_Logical_AND.obj");


            //merge all of the models together
            var poser = pose1.MergeWith(pose2)
                .MergeWith(pose3)
                .MergeWith(pose4)
                .MergeWith(pose5);

            poser.ExportOBJ("output/merged_all.obj");
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
