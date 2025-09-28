using System;
using OpenCvSharp;

namespace UnifiedForensicsAnalyze.Features
{


    public class UnifiedAnalyzer
    {

        private readonly ImageObject _objImage;
        private readonly Queue<Func<Mat, Mat>> _pipeLineStages;
        private readonly List<String> _stageName;
        private readonly Dictionary<String, Mat> _pipeLineResults;


        public UnifiedAnalyzer(ImageObject objImage)
        {
            _objImage = objImage ?? throw new ArgumentNullException(nameof(objImage));

            _pipeLineStages = new Queue<Func<Mat, Mat>>();
            _stageName = new List<string>();
            _pipeLineResults = new Dictionary<string, Mat>();
        }



        public void RunAnalysis()
        {
            Console.WriteLine("Starting Unified Forensics Analysis....");

            Mat current = _objImage.PrepImage();
            _objImage.SaveTemp(current, "prepProcess.png");
            // _pipeLineResults["PrepImage"] = current.Clone();

            // int stageIndex = 0;

            // while (_pipeLineStages.Count > 0)
            // {
            //     Func<Mat, Mat> stage = _pipeLineStages.Dequeue();
            //     string stageName = _stageName[stageIndex];

            //     current = stage(current);
            //     _pipeLineResults[stageName] = current.Clone();

            //     _objImage.SaveTemp(current, $"{stageIndex + 1}_{stageName}.png");

            //     stageIndex++;
            // }
        }



        public void RegisterPipeline(string name, Func<Mat, Mat> stage)
        {
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("Stage name cannot be empty! ", nameof(name));

            _pipeLineStages.Enqueue(stage ?? throw new ArgumentNullException(nameof(stage)));
            _stageName.Add(name);

        }




        public Mat? GetResult(string stageName)
        {
            return _pipeLineResults.ContainsKey(stageName) ? _pipeLineResults[stageName] : null;
        }


    }






}