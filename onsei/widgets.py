import ipywidgets as widgets
from traitlets import Dict


class SampleSelector(widgets.VBox):
    value = Dict().tag(sync=True)

    def __init__(self, samples, **kwargs):
        super().__init__(**kwargs)

        self.samples = samples

        self.collection = list(self.samples.keys())[0]
        self.sentence = list(samples[self.collection].keys())[0]

        self.value = self.selected_sample()

        self.collection_selector = widgets.Dropdown(
            options=samples.keys(),
            value=self.collection,
            description='Sample collection:',
            disabled=False,
            layout=widgets.Layout(width='100%'),
            )
        self.collection_selector.observe(self._update_collection, 'value')

        self.sentence_selector = widgets.Dropdown(
            options=samples[self.collection].keys(),
            value=self.sentence,
            description='Sentence:',
            disabled=False,
            layout=widgets.Layout(width='100%'),
            )
        self.sentence_selector.observe(self._update_sentence, 'value')

        self.children = [self.collection_selector, self.sentence_selector]

    def selected_sample(self):
        return self.samples[self.collection][self.sentence]

    def _update_collection(self, change):
        self.collection = change["new"]
        sentences = list(self.samples[self.collection].keys())
        with self.sentence_selector.hold_sync():
            self.sentence_selector.options = sentences
            self.sentence_selector.value = sentences[0]
        self.value = self.selected_sample()

    def _update_sentence(self, change):
        self.sentence = change["new"]
        self.value = self.selected_sample()
