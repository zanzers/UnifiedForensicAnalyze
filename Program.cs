using System;
using UnifiedForensicsAnalyze;
using UnifiedForensicsAnalyze.Features;


namespace UnifiedForensicsAnalyze
{

    class Program
    {

        public static void Main(string[] args)
        {
           try
            {
                Entry core = new Entry();
                string uploadDir = "uploads";
                string extractedDir = Path.Combine("ExtractedData");

                if (!Directory.Exists(uploadDir))Directory.CreateDirectory(uploadDir);

                if (Directory.Exists(extractedDir))Directory.Delete(extractedDir, true);
        
                Directory.CreateDirectory(extractedDir);
                

                string[] files = Directory.GetFiles(uploadDir);
                string[] subDirs = Directory.GetDirectories(uploadDir);

                if (subDirs.Length > 0)
                {
                    core.bInput(uploadDir);
                }
                else if (files.Length > 0)
                {
            
                    string path = files[0];
                    string ext = Path.GetExtension(path).ToLower();
                    string destPath = Path.Combine(extractedDir, "input" + ext);
                    File.Copy(path, destPath, true);
                    
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                    {
                        Console.WriteLine("[INFO] Detected image input.");
                        core.sInput(path);
                    }
                    else if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv")
                    {
                        core.vInput(destPath);
                    }
                }
                else
                {
                    throw new Exception("[FATAL] No files found in uploads folder.");
                }     

                

                Console.WriteLine("Execution complete.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FATAL] {ex.Message}");
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine(ex.Message);
                Console.ResetColor();
            }
        }
        
    }

}
