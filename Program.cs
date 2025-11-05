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
                    core.bInput(uploadDir);
                }
                else if (files.Length > 0)
                {
                    string path = files[0]; 
                    core.sInput(path);
                }
                else
                {
                    throw new Exception("[FATAL] No files found in uploads folder.");
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
