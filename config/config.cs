using System.Collections.Generic;
using UnifiedForensicsAnalyze.Pipeline;
using UnifiedForensicsAnalyze.Features;

namespace UnifiedForensicsAnalyze.Pipeline
{
    public class Config
    {
        public string Input { get; set; } = "";
        public string OutputPath { get; set; } = "ExtractedData";


        public bool FeaturesEnable { get; set; } = true;
        public bool MLEnabled { get; set; } = true;

        public List<string> SelectedStages { get; set; } = new List<string>();
        public List<IAnalysisStage> ExtraStages { get; set; } = new List<IAnalysisStage>();
    }
}
