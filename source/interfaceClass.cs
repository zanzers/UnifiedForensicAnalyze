using System;
using OpenCvSharp;
using Features_Write;

namespace UnifiedForensicsAnalyze.Features
{
    public interface IAnalysisStage
    {
        string Name { get; }
        StageResult Process(Mat input);
    }

    public class StageResult
    {
        public Mat? OutputImage { get; set; }
        public Dictionary<string, double>? Features { get; set; }

        public string? Output {get; set;}
        public bool Success {get; set;}
    }

    public class SaveFeatures
    {
        public static void HandleSave(UnifiedAnalyzer.InputCaller callerType, Dictionary<string, double> combinedFeatures, string? label = null)
        {
            if (combinedFeatures == null || combinedFeatures.Count == 0)
            {
                Console.WriteLine("[WARN] No features to save.");
                return;
            }

            try
            {
                switch (callerType)
                {
                    case UnifiedAnalyzer.InputCaller.bInput:
                        FeatureExcelWriter.AppendFeatures(combinedFeatures, label);
                        Console.WriteLine("[INFO] All stage features successfully appended to Excel.\n");
                        break;

                    default:
                        // if (result.OutputImage != null)
                        //     _objImage.SaveTemp(result.OutputImage, $"{stage.Name}_result.png");

                        FeatureJsonWriter.AppendFeatures(combinedFeatures);
                        Console.WriteLine("[INFO] Features successfully saved to JSON.\n");
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to save features: {ex.Message}");
            }
        }
    }
}
