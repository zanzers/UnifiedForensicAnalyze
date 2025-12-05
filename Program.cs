using System;
using UnifiedForensicsAnalyze;
using UnifiedForensicsAnalyze.Features;
using UnifiedForensicsAnalyze.Pipeline;

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
                
                var config = new Config
                {
                    Input = uploadDir,
                    OutputPath = extractedDir,
                    FeaturesEnable = true,
                    MLEnabled = false,
                    SelectedStages = new List<string>(),
                    ExtraStages = new List<IAnalysisStage>()
                };

                var runner  = new PipelineRun(config);
                runner.Run();

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
