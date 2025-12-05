using System;
using System.IO;
using OpenCvSharp;
using System.Diagnostics;

namespace UnifiedForensicsAnalyze.Features
{

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

        public static void CleanUp(string path = "uploads")
        {
            try
            {
                if (!Directory.Exists(path)) return;

                foreach (string file in Directory.GetFiles(path))
                {
                    File.Delete(file);
                }

                foreach (string dir in Directory.GetDirectories(path))
                {
                    Directory.Delete(dir, true);
                }
                Console.WriteLine($"[INFO] Upload folder cleaned: {path}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to clean uploads folder: {ex.Message}");
            }
        }

        public void Dispose()
        {
            InputImage?.Dispose();
        }


    
    }





    public static class PythonRunner
    {

        public static string Run(string scriptName,string inputType = "s", params string[] args)
        {

            // Deployment setting

            // string pyExe = Path.Combine("_mmmn", "Py", ".venv", "Scripts", "python.exe");
            // string scriptPath = Path.Combine("_mmmn","Py", "ML", scriptName);

            string pyExe = Path.Combine("Py", ".venv", "Scripts", "python.exe");
            
            string scriptPath;
            if(inputType == "s"){scriptPath = Path.Combine("Py", "ML", scriptName);}
            else{scriptPath = Path.Combine("Py", "ML", "Traning", scriptName);}


            string argString = string.Join(" ", args.Select(a => $"\"{a}\""));
            if (!File.Exists(scriptPath)){Console.WriteLine("Python script not found at: " + Path.GetFullPath(scriptPath));return "Script not found";}
            Console.WriteLine($">>> Running Python script: {scriptName}");

            
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = pyExe,
                Arguments = $"\"{scriptPath}\" {argString}",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using (Process process = new Process { StartInfo = psi })
            {
                process.Start();

                string output = process.StandardOutput.ReadToEnd();
                string errors = process.StandardError.ReadToEnd();

                process.WaitForExit();

                if (!string.IsNullOrWhiteSpace(errors))
                    Console.WriteLine("Python error: " + errors);


                Console.WriteLine("[PYTHON OUTPUT] " + output);

                return output.Trim();
            }
        }
    }



    public static class Bypass
    {
        private static string _storedPath;


        public static string ByPass(string imgPath)
        {
       
            _storedPath = imgPath;
            return _storedPath;
        }

          public static string GetPath()
        {
            return _storedPath;
        }

        public static void SaveOriginal(string uploadDir = "uploads")
        {
            string[] files = Directory.GetFiles(uploadDir);
            string extractedDir = Path.Combine("ExtractedData");

            if (files.Length == 0)
                throw new Exception("[ERROR] No files found in uploads folder.");
            string srcFile = files[0];
            string ext = Path.GetExtension(srcFile).ToLower();
            string destPath = Path.Combine(extractedDir, "input" + ext);
            File.Copy(srcFile, destPath, true);

            Console.WriteLine($"[INFO] Saved original to: {destPath}");
        }



    }

}
