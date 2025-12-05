using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Spreadsheet;
using System.Text.Json;

namespace Features_Write
{
    public static class FeatureExcelWriter
    {

        private static readonly string baseFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\Py\ML\Py/ML/Traning");
        private static readonly string filePath = Path.Combine(baseFolder, "features_dataset.xlsx");
        private static bool isFirstWrite = true;

        public static void AppendFeatures(Dictionary<string, double> features, string? label = null)
        {
            if (features == null || features.Count == 0)
            {
                Console.WriteLine("[WARN] No features to save in Excel.");
                return;
            }

            try
            {
                // Ensure the folder exists
                Directory.CreateDirectory(baseFolder);

                string[] featureOrder = new string[]
                                {
                                    "Label",
                                    "Ela_Entropy",
                                    "Ela_Mean",
                                    "Ela_StdDev",
                                    "Ela_Kurtosis",
                                    "Ela_Skewness",
                                    "PRNU_GMean",
                                    "PRNU_GStdDev",
                                    "PRNU_GEnergy",
                                    "PRNU_BMean",
                                    "PRNU_BStd",
                                    "PRNU_BEnergy",
                                    "PRNU_BMax",
                                    "PRNU_B_Std_Max",
                                    "Wavelet_Energy_LL",
                                    "Wavelet_Energy_HF",
                                    "Wavelet_HF_to_LL_Ratio",
                                    "PRNU_Normal_Mean",
                                    "SVD_Top1",
                                    "SVD_stdSV",
                                    "SVD_Mean",
                                    "SVD_Top5",
                                    "SVD_Entropy",
                                    "SVD_Top10",
                                    "IWT_HL_Var",
                                    "IWT_HH_Entropy",
                                    "IWT_LL_Entropy",
                                    "IWT_LH_Mean",
                                    "IWT_LH_Var",
                                    "IWT_HL_Mean",
                                    "IWT_HH_Var",
                                    "IWT_HL_Entropy",
                                    "IWT_LH_Entropy",
                                    "IWT_LL_Mean",
                                    "IWT_HH_Mean",
                                    "IWT_LL_Var",
                                    "CNN_Confidence",
                                    "CNN_Prob_1",
                                    "CNN_Prob_0",
                                    "CNN_Label",
                                    "CNN_Prob_2",
                                };


                if (isFirstWrite || !File.Exists(filePath))
                {
                    if (File.Exists(filePath)) File.Delete(filePath);

                    Console.WriteLine($"[INFO] Creating new Excel file at: {filePath}");

                    using (SpreadsheetDocument document = SpreadsheetDocument.Create(filePath, SpreadsheetDocumentType.Workbook))
                    {
                        WorkbookPart workbookPart = document.AddWorkbookPart();
                        workbookPart.Workbook = new Workbook();
                        WorksheetPart worksheetPart = workbookPart.AddNewPart<WorksheetPart>();
                        SheetData sheetData = new SheetData();
                        worksheetPart.Worksheet = new Worksheet(sheetData);

                        Sheets sheets = workbookPart.Workbook.AppendChild(new Sheets());
                        sheets.Append(new Sheet()
                        {
                            Id = workbookPart.GetIdOfPart(worksheetPart),
                            SheetId = 1,
                            Name = "Features"
                        });

                        // Header row
                        Row headerRow = new Row();
                        foreach (var key in featureOrder)
                            headerRow.Append(CreateCell(key));
                        sheetData.Append(headerRow);



                        // First data row
                        Row dataRow = new Row();
                        foreach (var key in featureOrder)
                        {
                            string value;
                            if (key == "Label")
                                value = label ?? "";
                            else
                                value = features.ContainsKey(key) ? features[key].ToString("F6") : "0";

                            // Append the cell
                            dataRow.Append(CreateCell(value));

                            // Debug print
                            Console.WriteLine($"Writing to Excel - Column: {key}, Value: {value}");
                        }
                        sheetData.Append(dataRow);

                        workbookPart.Workbook.Save();

                    }

                    isFirstWrite = false;
                }
                else
                {
                    using (SpreadsheetDocument document = SpreadsheetDocument.Open(filePath, true))
                    {
                        var workbookPart = document.WorkbookPart!;
                        var sheet = workbookPart.Workbook.Sheets!.GetFirstChild<Sheet>()!;
                        var worksheetPart = (WorksheetPart)workbookPart.GetPartById(sheet.Id!);
                        SheetData sheetData = worksheetPart.Worksheet.GetFirstChild<SheetData>()!;

                        // Add new row for this image
                        Row newRow = new Row();
                        foreach (var key in featureOrder)
                        {
                            string value = (key == "Label") ? (label ?? "") : (features.ContainsKey(key) ? features[key].ToString("F6") : "0");
                            newRow.Append(CreateCell(value));

                            // Debug print
                            Console.WriteLine($"Writing to Excel - Column: {key}, Value: {value}");
                        }
                        sheetData.Append(newRow);
                        worksheetPart.Worksheet.Save();
                    }
                }

                Console.WriteLine($"[INFO] Features successfully appended to Excel: {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to write Excel file: {ex.Message}");
            }
        }

        private static Cell CreateCell(string text)
        {
            return new Cell()
            {
                DataType = CellValues.String,
                CellValue = new CellValue(text)
            };
        }


       

    }

    public static class FeatureJsonWriter
    {
    


        private static readonly string PredictionjsonPath = Path.Combine("ExtractedData", "Output");
        private static readonly string[] featureOrder = new string[]
        {
                                    "Label",
                                    "Ela_Entropy",
                                    "Ela_Mean",
                                    "Ela_StdDev",
                                    "Ela_Kurtosis",
                                    "Ela_Skewness",
                                    "PRNU_GMean",
                                    "PRNU_GStdDev",
                                    "PRNU_GEnergy",
                                    "PRNU_BMean",
                                    "PRNU_BStd",
                                    "PRNU_BEnergy",
                                    "PRNU_BMax",
                                    "PRNU_B_Std_Max",
                                    "Wavelet_Energy_LL",
                                    "Wavelet_Energy_HF",
                                    "Wavelet_HF_to_LL_Ratio",
                                    "PRNU_Normal_Mean",
                                    "SVD_Top1",
                                    "SVD_stdSV",
                                    "SVD_Mean",
                                    "SVD_Top5",
                                    "SVD_Entropy",
                                    "SVD_Top10",
                                    "IWT_HL_Var",
                                    "IWT_HH_Entropy",
                                    "IWT_LL_Entropy",
                                    "IWT_LH_Mean",
                                    "IWT_LH_Var",
                                    "IWT_HL_Mean",
                                    "IWT_HH_Var",
                                    "IWT_HL_Entropy",
                                    "IWT_LH_Entropy",
                                    "IWT_LL_Mean",
                                    "IWT_HH_Mean",
                                    "IWT_LL_Var",
                                    "CNN_Confidence",
                                    "CNN_Prob_1",
                                    "CNN_Prob_0",
                                    "CNN_Label",
                                    "CNN_Prob_2",
        };



        public static string GetJsonpath(string type = "features")
        {
            if (!Directory.Exists(PredictionjsonPath))
                Directory.CreateDirectory(PredictionjsonPath);
                
            return type == "features"
                ? Path.Combine(PredictionjsonPath, "features.json")
                : Path.Combine(PredictionjsonPath, "classification_result.json");
        }


        public static void AppendFeatures(Dictionary<string, double> features, string type = "features")
            {
                if (features == null || features.Count == 0)
                {
                    Console.WriteLine("[WARN] No features to save in JSON.");
                    return;
                }

                try
                {
                  
                    string jsonPath = GetJsonpath(type);
                    string folder = Path.GetDirectoryName(jsonPath)!;

                    if (Directory.Exists(jsonPath))
                    {
                        Console.WriteLine($"[WARN] Found a directory with same name as file: {jsonPath}. Deleting it...");
                        Directory.Delete(jsonPath, true);
                    }

                    if (!Directory.Exists(folder))
                        Directory.CreateDirectory(folder);

                    if (File.Exists(jsonPath))
                        File.Delete(jsonPath);

                
                    Dictionary<string, double> orderedFeatures = new();
                    foreach (var key in featureOrder)
                    {
                        orderedFeatures[key] = features.ContainsKey(key) ? features[key] : 0;
                        Console.WriteLine($"JSON - Key: {key}, Value: {orderedFeatures[key]}");
                    }

                    List<Dictionary<string, double>> allFeatures = new() { orderedFeatures };

                    string jsonContent = JsonSerializer.Serialize(allFeatures, new JsonSerializerOptions
                    {
                        WriteIndented = true
                    });

                    File.WriteAllText(jsonPath, jsonContent);

                    Console.WriteLine($"[INFO] Features successfully appended to JSON: {jsonPath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ERROR] Failed to write JSON file: {ex.Message}");
                }
            }

    }

}

