using System;
using OpenCvSharp;

namespace UnifiedForensicsAnalyze.Features
{


    public class UnifiedAnalyzer
    {

        private readonly ImageObject _objImage;
        private readonly Queue<IAnalysisStage> _stage;
        private readonly Dictionary<String, StageResult> _pipeLineResults;


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
                    Console.WriteLine($"[{stage.Name}] Features: {string.Join(", ", result.Features)}");
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
        public List<double>? Features { get; set; }
    }
}