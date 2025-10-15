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
                string Path = "datasets/2/ai_1.jpg";

                using (ImageObject imgObj = new ImageObject(Path))
                {
                    UnifiedAnalyzer analyzer = new UnifiedAnalyzer(imgObj);


                    analyzer.AddStage(new ELAStage());
                    analyzer.AddStage(new SVDStage());
                    analyzer.AddStage(new IWTStage());
                    analyzer.AddStage(new PRNUStage());

                    analyzer.RunAnalysis();
                    
                    Console.WriteLine("Preprocessing Complete")
;
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }

}
