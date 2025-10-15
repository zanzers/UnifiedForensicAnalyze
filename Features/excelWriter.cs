using System;
using System.IO;
using System.Collections.Generic;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Spreadsheet;

namespace WriteExcel
{
    public static class FeatureExcelWriter
    {
        private static readonly string filePath = "features.xlsx";
        private static bool isFirstWrite = true;

        public static void AppendFeatures(Dictionary<string, double> features)
        {
            if (isFirstWrite || !File.Exists(filePath))
            {
                if (File.Exists(filePath))
                    File.Delete(filePath);
                    Console.WriteLine("Creating new Excel file.");

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


                    Row headerRow = new Row();
                    Row dataRow = new Row();




                    foreach (var kv in features)
                    {
                        headerRow.Append(CreateCell(kv.Key));
                        dataRow.Append(CreateCell(kv.Value.ToString("F6")));
                    }

                    sheetData.Append(headerRow);
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

                    // Get header and data rows (should already exist)
                    Row headerRow = sheetData.Elements<Row>().FirstOrDefault()!;
                    Row dataRow = sheetData.Elements<Row>().Skip(1).FirstOrDefault()!;



                    // Append new stage's features as new columns
                    foreach (var kv in features)
                    {
                        headerRow.Append(CreateCell(kv.Key));
                        dataRow.Append(CreateCell(kv.Value.ToString("F6")));
                    }

                    worksheetPart.Worksheet.Save();
                }
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
}
