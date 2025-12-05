using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnifiedForensicsAnalyze.Features;

namespace UnifiedForensicsAnalyze.Pipeline
{
    public class Pipeline
    {
        private readonly Config _config;
        public Pipeline(Config config)
        {
            _config = config;
        }


        public void Execute()
        {
            Console.WriteLine("===== MMMN Pipeline Starting =====");
            Console.WriteLine($"Input Image     : {_config.Input}");
            Console.WriteLine($"Output Path     : {_config.OutputPath}");

            Entry core = new Entry();
            var inputActions = new (Func<string, bool> Condition, Action<string> Action)[]
            {
                (InputType.DirectoryContainsSubfolders, path => InputRunner.RunDataset(path, core)),
                (InputType.DirectoryContainsImages, path => InputRunner.RunSingleImage(InputType.GetFirstImage(path), core)),
                (InputType.DirectoryContainsVideo, path => InputRunner.RunVideo(InputType.GetFirstVideo(path), core))
            };

            bool handled = false;
            foreach (var (condition, action) in inputActions)
            {
                if (condition(_config.Input))
                {
                    action(_config.Input);
                    handled = true;
                    break;
                }
            }

            if (!handled)
            {
                Console.WriteLine("[ERROR] No valid input found.");
                return;
            }

            if (File.Exists(_config.Input) || InputType.DirectoryContainsImages(_config.Input))
            {
                string imgPath = File.Exists(_config.Input) ? _config.Input : InputType.GetFirstImage(_config.Input);

                using (ImageObject imgObj = new ImageObject(imgPath))
                {
                    UnifiedAnalyzer analyzer = new UnifiedAnalyzer(imgObj);
                    analyzer.CallerInput(UnifiedAnalyzer.InputCaller.sInput);

                    if (_config.FeaturesEnable)
                    {
                        var featureStages = _config.SelectedStages.Count > 0
                            ? _config.SelectedStages
                            : new List<string> { "ELA", "SVD", "IWT", "PRNU", "RF"};

                        foreach (string layerName in featureStages)
                        {
                            IAnalysisStage? layer = LayerFactory.Create(layerName);
                            if (layer != null)
                            {
                                Console.WriteLine($" -> Adding Stage: {layerName}");
                                analyzer.AddStage(layer);
                            }
                            else
                            {
                                Console.WriteLine($"[WARNING] Stage not found: {layerName}");
                            }
                        }
                    }

                    if (_config.MLEnabled)
                    {
                        IAnalysisStage? mlStage = LayerFactory.Create("CNN");
                        if (mlStage != null)
                        {
                            Console.WriteLine(" -> Adding ML Stage: CNN");
                            analyzer.AddStage(mlStage);
                        }
                    }

                    foreach (var stage in _config.ExtraStages)
                    {
                        Console.WriteLine($" -> Adding Custom Stage: {stage.Name}");
                        analyzer.AddStage(stage);
                    }

                    analyzer.RunAnalysis();
                }
            }

            Console.WriteLine("===== MMMN Pipeline Finished =====");
        }
    }

    public class PipelineRun
    {
        private readonly Pipeline _pipeline;

        public PipelineRun(Config config)
        {
            _pipeline = new Pipeline(config);
        }

        public void Run()
        {
            Console.WriteLine("[NOTICE]: Starting pipeline...");
            _pipeline.Execute();
            Console.WriteLine("[NOTICE]: Pipeline finished.");
        }
    }

    public static class LayerFactory
    {
        public static IAnalysisStage? Create(string name)
        {
            switch (name.Trim().ToLower())
            {
                case "ela":return new ELAStage();
                case "svd":return new SVDStage();
                case "iwt":return new IWTStage();
                case "prnu":return new PRNUStage();
                case "cnn":return new CnnStage();
                default:
                    Console.WriteLine($"[LayerFactory] Unknown stage: {name}");
                    return null;
            }
        }
    }

    public static class InputType
    {
        private static readonly string[] ImageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
        private static readonly string[] VideoExtensions = { ".mp4", ".avi", ".mov", ".mkv" };

        public static bool DirectoryContainsSubfolders(string path) =>
            Directory.Exists(path) && Directory.GetDirectories(path).Length > 0;

        public static bool DirectoryContainsImages(string path) =>
            Directory.Exists(path) && Directory.GetFiles(path)
                .Any(f => ImageExtensions.Contains(Path.GetExtension(f).ToLower()));

        public static bool DirectoryContainsVideo(string path) =>
            Directory.Exists(path) && Directory.GetFiles(path)
                .Any(f => VideoExtensions.Contains(Path.GetExtension(f).ToLower()));

        public static string GetFirstImage(string path) =>
            Directory.GetFiles(path)
                     .First(f => ImageExtensions.Contains(Path.GetExtension(f).ToLower()));

        public static string GetFirstVideo(string path) =>
            Directory.GetFiles(path)
                     .First(f => VideoExtensions.Contains(Path.GetExtension(f).ToLower()));
    }

    public static class InputRunner
    {
        public static void RunDataset(string path, Entry core) => core.bInput(path);
        public static void RunSingleImage(string imgPath, Entry core) => core.sInput(imgPath);
        public static void RunVideo(string videoPath, Entry core) => core.vInput(videoPath);
    }
}
