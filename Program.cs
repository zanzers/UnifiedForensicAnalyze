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

                string path = "datasets/0/or_1.jpg";
                string path1 = "datasets";
                // Bacth input name: bInput;
            
                Entry core = new Entry();


                core.bInput(path1);
                // core.sInput(path);

                Console.WriteLine("Execution complete.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FATAL] {ex.Message}");
            }
        }
        
    }

}
