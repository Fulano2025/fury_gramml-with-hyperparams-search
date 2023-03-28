from gramml.constants import *
from itertools import count


class PipelineInitializer():
    """
    Pipeline Initializer que utilizará el dispatcher para instanciar los pipelines
    """

    def initialize_pipeline(self, pipeline_blueprint, node_id):
        """
        Método recursivo que navega el arból de definición del pipeline y retorna un string para instanciar
        """
        
        if NODE_STEPS in pipeline_blueprint.keys(): 
            arguments = ', '.join([self.initialize_pipeline(step, node_id) for step in pipeline_blueprint[NODE_STEPS]])
            return "(\'"+str(next(node_id))+"\', "+pipeline_blueprint[NODE_CLASS]+f'([{arguments}])'+")"
        else:
            params = []
            if NODE_HYPERPARAMS in pipeline_blueprint.keys():         
                for h in pipeline_blueprint[NODE_HYPERPARAMS]:
                    print()
                    param = f'{h[HP_NAME]}="{h[HP_VALUE]}"' if type(h[HP_VALUE])==str else f"{h[HP_NAME]}={h[HP_VALUE]}" 
                    params.append(param)
            
            transformer = pipeline_blueprint[NODE_CLASS]+f"({', '.join(params)})"

            if NODE_COL_INDEX in pipeline_blueprint.keys():
                transformer_pipe =  "sklearn.compose.make_column_transformer(("+transformer+","+ str(pipeline_blueprint[NODE_COL_INDEX]) +"))"
            else:
                transformer_pipe = transformer
            return "(\'"+str(next(node_id))+"\', "+ transformer_pipe + ")"
    
    
    def get_pipeline(self, pipeline_blueprint):
        node_id = count(0)
        pipeline = self.initialize_pipeline(pipeline_blueprint, node_id)
        for import_ in IMPORTS:
            exec(import_)
    
        return eval(pipeline)