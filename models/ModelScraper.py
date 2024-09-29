from collections import deque


class ModelScraper():

    def __init__(self, model):


        self.model = model

    def scrape_model(self,retrieval_list):
        self.retrieval_list = retrieval_list
        # initialize output dict to have keys in retrieval dict with empty dict inside
        self.output_dict = {key: {} for key in retrieval_list}
        print(self.output_dict)

        next_layers = self.model.traversable_layers
        queue = deque(next_layers)

        while len(queue) > 0:
            layer = queue.popleft()

            if layer.name in self.output_dict.keys():

                input = layer.get_input()
                if input is not None:
                    self.output_dict[layer.name]["input"] = input

                output = layer.get_output()
                if output is not None:
                    self.output_dict[layer.name]["output"] = output


            queue.extend(layer.traversable_layers)


