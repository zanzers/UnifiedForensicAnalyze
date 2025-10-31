using System;
using System.IO;
using OpenCvSharp;

namespace UnifiedForensicsAnalyze.Features
{

    // Task: Handle loading, storing, and basic transformations!

    public class ImageObject : IDisposable
    {

        public string FilePath { get; private set; }
        public Mat InputImage { get; private set; }

        private static string OutputFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Output");
        private static int SaveCounter = 0;


        public ImageObject(string filePath)
        {
            FilePath = string.IsNullOrWhiteSpace(filePath) ? throw new ArgumentException("Invalid filePath: ", nameof(filePath)) : filePath;
            InputImage = Cv2.ImRead(FilePath, ImreadModes.Color);

            InputImage = InputImage.Empty() ? throw new Exception($"Failed to load image: {FilePath}") : InputImage;
           
            if (!Directory.Exists(OutputFolder)) Directory.CreateDirectory(OutputFolder);
            else
                foreach (string file in Directory.GetFiles(OutputFolder))
                    File.Delete(file);


        }

        

        public Mat PrepImage()
        {
            Console.WriteLine("Preprocessing image...");

            Mat gray = ToGray();
            Mat normalized = Normalize(gray);

            return normalized;
        }


        public Mat ToGray()
        {
            Mat gray = new Mat();
            Cv2.CvtColor(InputImage, gray, ColorConversionCodes.BGR2GRAY);
            return gray;
        }

        public Mat Normalize(Mat source, double alpha = 0, double beta = 1)
        {
            Mat normalized = new Mat();
            Cv2.Normalize(source, normalized, alpha, beta, NormTypes.MinMax, MatType.CV_32F);
            return normalized;
        }


        public void SaveTemp(Mat? imageToSave = null, string? fileName = null)
        {
            Mat toSave = imageToSave ?? InputImage;
            
            if (toSave.Empty())
            {
                throw new Exception("No image data available to save.");
            }

            SaveCounter++;
            string finalName = fileName ?? $"result_{SaveCounter}.png";
            string outputPath = Path.Combine(OutputFolder, finalName);

            Mat saveReady = new Mat();
            if (toSave.Depth() != MatType.CV_8U)
            {
                toSave.ConvertTo(saveReady, MatType.CV_8U, 255.0);
            }
            else
            {
                saveReady = toSave;
            }

            Cv2.ImWrite(outputPath, saveReady);

            
        }

        public void Dispose()
        {
            InputImage?.Dispose();
        }
       

    }



    


}