using System;
using OpenCvSharp;
using System.Collections.Concurrent;



namespace UnifiedForensicsAnalyze.Features
{
    public class UnifiedAnalyzer
    {
        private readonly ImageObject? _objImage;
        private string? _videoPath;
        private readonly Queue<IAnalysisStage> _stage;
        private readonly Dictionary<string, StageResult> _pipeLineResults;
        public enum InputCaller { bInput, sInput}

        private InputCaller _callerType;
        private string? _label;

        public UnifiedAnalyzer(ImageObject objImage)
        {
            _objImage = objImage ?? throw new ArgumentNullException(nameof(objImage));
            _stage = new Queue<IAnalysisStage>();
            _pipeLineResults = new Dictionary<string, StageResult>();
        }

        public UnifiedAnalyzer(string videoPath)
        {
            if (string.IsNullOrWhiteSpace(videoPath))
                throw new ArgumentNullException(nameof(videoPath));

            _videoPath = videoPath;
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

            if(_objImage != null)
            {
                Mat original = _objImage.InputImage.Clone();
                RunStage(original);
            }
            else if(_videoPath != null)
            {
                RunStage(null);
            }
            else
            {
                throw new Exception("No valid input provided for UnifiedAnalyzer: neither image nor video.");

            }
            
        }

    // QUEU and DEQUE Process
    //    public void RunStage(Mat? image)
    //     {
    //         Dictionary<string, double> combinedFeatures = new Dictionary<string, double>();

    //         while (_stage.Count > 0)
    //         {
    //             IAnalysisStage stage = _stage.Dequeue();

    //             StageResult result;
    //             if (image != null)
    //             {
    //                 result = stage.Process(image); 
    //             }
    //             else
    //             {
    //                 result = stage.Process(null);  
    //             }

    //             if (result.Features != null && result.Features.Count > 0)
    //             {
    //                 foreach (var kvp in result.Features)
    //                 {
    //                     combinedFeatures[kvp.Key] = kvp.Value;
    //                 }
    //             }
    //         }

    //         SaveFeatures.HandleSave(_callerType, combinedFeatures, _label);
    //         Console.WriteLine("Analysis complete!");
    //     }

        public void RunStage(Mat? image)
        {
            var combinedFeatures = new ConcurrentDictionary<string, double>();
            var stages = _stage.ToList();

            // Create tasks for each stage
            var tasks = stages.Select(stage => Task.Run(() =>
            {
                StageResult result = stage.Process(image);

                if (result.Features != null && result.Features.Count > 0)
                {
                    foreach (var kvp in result.Features)
                    {
                        // Use ConcurrentDictionary to avoid threading issues
                        combinedFeatures[kvp.Key] = kvp.Value;
                    }
                }

                return result; // optional, if you want to keep per-stage results
            })).ToArray();

            // Wait for all tasks to finish
            Task.WaitAll(tasks);

            // Save combined features as usual
            SaveFeatures.HandleSave(_callerType, combinedFeatures.ToDictionary(k => k.Key, v => v.Value), _label);

            Console.WriteLine("All stages completed in parallel!");
        }

       


        public StageResult? GetResult(string stageName)
        {
            return _pipeLineResults.ContainsKey(stageName) ? _pipeLineResults[stageName] : null;
        }
    }
}
