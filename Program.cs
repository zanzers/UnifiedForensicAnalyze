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
                string Path = "datasets/2/ai_4.jpg";

                using (ImageObject imgObj = new ImageObject(Path))
                {
                    UnifiedAnalyzer analyzer = new UnifiedAnalyzer(imgObj);


                    analyzer.AddStage(new PRNUStage());
                    analyzer.AddStage(new ELAStage());
                    analyzer.AddStage(new SVDStage());
                    analyzer.AddStage(new IWTStage());

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
