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
 
                string path = "datasets/1/tam_(9).jpg";
                // string path1 = "datasets";

            

                // core.bInput(path1);
                core.sInput(path);

                Console.WriteLine("Execution complete.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FATAL] {ex.Message}");
            }
        }
        
    }

}
