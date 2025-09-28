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
                string Path = "datasets/0/au_7.jpg";

                using (ImageObject imgObj = new ImageObject(Path))
                {
                    UnifiedAnalyzer analyzer = new UnifiedAnalyzer(imgObj);

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
