using System;
using OpenCvSharp;


namespace UnifiedForensicsAnalyze.Features
{
    public class UnifiedAnalyzer
    {
        private readonly ImageObject _objImage;
        private readonly Queue<IAnalysisStage> _stage;
        private readonly Dictionary<string, StageResult> _pipeLineResults;
        public enum InputCaller { bInput, sInput }

        private InputCaller _callerType;
        private string? _label;

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

        public void CallerInput(InputCaller caller, string? label = null)
        {
            _callerType = caller;
            _label = label;
        }

        public void RunAnalysis()
        {

            Mat original = _objImage.InputImage.Clone();
            Dictionary<string, double> combinedFeatures = new Dictionary<string, double>();

            while (_stage.Count > 0)
            {
                IAnalysisStage stage = _stage.Dequeue();
                StageResult result = stage.Process(original);

                if(result.Features != null && result.Features.Count > 0)
                {
                    foreach (var kvp in result.Features)
                    {
                        string key = $"{kvp.Key}";
                        combinedFeatures[key] = kvp.Value;
                    }
                }
            }

        

            SaveFeatures.HandleSave(_callerType, combinedFeatures, _label);

            Console.WriteLine("Analysis complete!");
        }

        public StageResult? GetResult(string stageName)
        {
            return _pipeLineResults.ContainsKey(stageName) ? _pipeLineResults[stageName] : null;
        }
    }
}
