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

                if (!Directory.Exists(uploadDir))
                {
                    Directory.CreateDirectory(uploadDir);
                    Console.WriteLine($"[INFO] Created upload directory at: {uploadDir}");
                }

                string[] files = Directory.GetFiles(uploadDir);
                string[] subDirs = Directory.GetDirectories(uploadDir);

                if (subDirs.Length > 0)
                {
                    Console.WriteLine("[INFO] Dataset mode detected...");
                    core.bInput(uploadDir);
                }
                else
                {
                    string path = files[0];
                    Console.WriteLine($"[INFO] Single file detected: {Path.GetFileName(path)}");
                    core.sInput(path);

                }       

                ImageObject.CleanUp("uploads");

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
