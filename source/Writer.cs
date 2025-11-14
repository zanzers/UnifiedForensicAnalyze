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

        private static readonly string baseFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\ExtractedData");
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
                        headerRow.Append(CreateCell("Label"));
                        foreach (var key in features.Keys)
                            headerRow.Append(CreateCell(key));
                        sheetData.Append(headerRow);

                        // First data row
                        Row dataRow = new Row();
                        dataRow.Append(CreateCell(label ?? ""));
                        foreach (var val in features.Values)
                            dataRow.Append(CreateCell(val.ToString("F6")));
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
                        newRow.Append(CreateCell(label ?? ""));
                        foreach (var val in features.Values)
                            newRow.Append(CreateCell(val.ToString("F6")));
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
    

        private static readonly string ExtractedjsonPath = Path.Combine("ExtractedData");
        private static readonly string PredictionjsonPath = Path.Combine("ExtractedData", "Output");

        public static string GetJsonpath(string type = "features")
        {

            string folder = type == "features" ? ExtractedjsonPath : PredictionjsonPath;
            if (!Directory.Exists(folder))
                Directory.CreateDirectory(folder);
                
            return type == "features"
                ? Path.Combine(folder, "features.json")
                : Path.Combine(folder, "classification_result.json");
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

       
            List<Dictionary<string, double>> allFeatures = new() { features };

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
