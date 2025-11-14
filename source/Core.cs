using System;
using OpenCvSharp;
using System.IO;
using UnifiedForensicsAnalyze;
using UnifiedForensicsAnalyze.Features;



namespace UnifiedForensicsAnalyze
{
    public class Entry
    {
        public void sInput(string imgPath)
        {
            // Validate path
            // Run UnifiedAnalyzer
            try
            {
                using (ImageObject imgObj = new ImageObject(imgPath))
                {
                    UnifiedAnalyzer analyzer = new UnifiedAnalyzer(imgObj);
                    analyzer.CallerInput(UnifiedAnalyzer.InputCaller.sInput);

                    analyzer.AddStage(new ELAStage());
                    analyzer.AddStage(new SVDStage());
                    analyzer.AddStage(new IWTStage());
                    analyzer.AddStage(new PRNUStage());
                    analyzer.AddStage(new CnnStage());

                    analyzer.RunAnalysis();

                    ImageObject.CleanUp("uploads");
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to write Excel: {ex.Message}");
                throw;
            }


        }

        public void bInput(string datasets)
        {
            // Loop through subfolders 0,1,2
            // Run UnifiedAnalyzer for each
            // Collect features + label
            // Append to Excel (features_dataset.xlsx)


            if (string.IsNullOrWhiteSpace(datasets))
            {
                throw new ArgumentNullException(nameof(datasets), "Dataset directory path cannot be null or empty.");
            }
            else if (!Directory.Exists(datasets))
            {
                throw new ArgumentNullException(nameof(datasets), $"Dataset folder not found: {datasets}");
            }

            string[] subDirs = Directory.GetDirectories(datasets);
            if (subDirs.Length == 0) throw new InvalidOperationException("No label subfolders (0, 1, 2, etc.) found inside dataset directory.");


            foreach (string subDir in subDirs)
            {
                string label = new DirectoryInfo(subDir).Name;
                string[] img = Directory.GetFiles(subDir, "*.*", SearchOption.TopDirectoryOnly);


                foreach (string imgPath in img)
                {
                    try

                    {
                        using (ImageObject imgObj = new ImageObject(imgPath))
                        {
                            UnifiedAnalyzer analyzer = new UnifiedAnalyzer(imgObj);
                            Console.WriteLine($"label: {label}");
                            analyzer.CallerInput(UnifiedAnalyzer.InputCaller.bInput, label);

                            analyzer.AddStage(new ELAStage());
                            analyzer.AddStage(new SVDStage());
                            analyzer.AddStage(new IWTStage());
                            analyzer.AddStage(new PRNUStage());
                            analyzer.AddStage(new CnnStage());

                            analyzer.RunAnalysis();


                        }

                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[ERROR] Failed to write Excel: {ex.Message}");
                        throw;
                    }


                }
            }



        }

        public void vInput(string videoPath)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(videoPath))
                    throw new ArgumentNullException(nameof(videoPath));

                UnifiedAnalyzer analyzer = new UnifiedAnalyzer(videoPath);

                analyzer.AddStage(new Extraction(videoPath));
                analyzer.RunAnalysis();
            }
            catch (Exception ex)
            {
                Console.WriteLine("[vInput] Error: " + ex.Message);
            }
        }

    }


}
