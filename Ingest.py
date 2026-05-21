from docling.document_converter import DocumentConverter

class loader:
    def __init__(self):
        self.converter=DocumentConverter()
    
    #Step1 Extract markdown Output
    def exportToMarkDownOutput(self, file_path:str):
        result=self.converter.convert(file_path)
        return result.document.export_to_markdown()