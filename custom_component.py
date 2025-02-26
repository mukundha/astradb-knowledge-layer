# from langflow.field_typing import Data
from langflow.custom import Component
from langflow.io import Output, FileInput
import json
from langflow.schema import Data

class CustomComponent(Component):
    display_name = "JSON array to List of Data"
    description = "Converts a JSON array to a list of Data objects."
    documentation: str = "http://docs.langflow.org/components/custom"
    icon = "code"
    name = "CustomComponent"

    inputs = [
        FileInput(name="fileInput",file_types=["json"],required=True,advanced=False),        
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> list[Data]:        
        with open(self.fileInput, "r") as f:
            content = f.read()            
            data = json.loads(content)            
            ret = [Data(text=d.get("chunk", ""), metadata=d.get("metadata", {})) for d in data]
        return ret
        
