using System;
using OpenCvSharp;
using WriteExcel;

namespace UnifiedForensicsAnalyze.Features
{

   
    public class UnifiedAnalyzer
    {
 
        private readonly ImageObject _objImage;
        private readonly Queue<IAnalysisStage> _stage;
        private readonly Dictionary<String, StageResult> _pipeLineResults;
        private readonly bool Caller;


        public UnifiedAnalyzer(ImageObject objImage)
        {
            _objImage = objImage ?? throw new ArgumentNullException(nameof(objImage));

            _stage = new Queue<IAnalysisStage>();
            _pipeLineResults = new Dictionary<string, StageResult>();
        }

        public void AddStage(IAnalysisStage stage)
        {
            _stage.Enqueue(stage);
        }



        public void RunAnalysis()
        {
            Console.WriteLine("Starting Unified Forensics Analysis....");

            Mat original = _objImage.InputImage.Clone();
    

            while (_stage.Count > 0)
            {
                IAnalysisStage stage = _stage.Dequeue();
                Console.WriteLine($"Running stage: {stage.Name}...");

                StageResult result = stage.Process(original);
                if (result.OutputImage != null)
                {
                    _objImage.SaveTemp(result.OutputImage, $"{stage.Name}_result.png");
                   
                }

                _pipeLineResults[stage.Name] = result;
               
                if (result.Features != null)
                {
                     result.WriteData();

                }
            }
            
            Console.WriteLine("Analysis complete!");
        }


        public StageResult? GetResult(string stageName)
        {
            return _pipeLineResults.ContainsKey(stageName) ? _pipeLineResults[stageName] : null;
        }


    }


// Interface Properties :
    public interface IAnalysisStage
    {
        string Name { get; }
        StageResult Process(Mat input);
    }

    public class StageResult
    {
        public Mat? OutputImage { get; set; }
        public Dictionary<string, double>? Features { get; set; }

        public void WriteData()
        {
            if (Features == null || Features.Count == 0)
            {
                Console.WriteLine("No features found.");
                return;
            }

            // Console.WriteLine("---- Feature Summary ----");
            // foreach (var kv in Features)
            //     Console.WriteLine($"{kv.Key}: {kv.Value:F6}");
            // Console.WriteLine("--------------------------");


            // true\false check the caller to add the folder_name, else add " " ;
            // true we continue to caller;
            try
            {
                FeatureExcelWriter.AppendFeatures(Features);
                Console.WriteLine("Features successfully appended to Excel.\n");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to write Excel file: {ex.Message}");
            }
        }
    }
}