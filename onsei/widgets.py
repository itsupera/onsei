import html
import logging
import os
import pathlib

import ipywidgets as widgets
from traitlets import Dict

from onsei.sentence import Sentence
from onsei.utils import check_wav_content_is_valid, convert_audio, \
    parse_wav_file_to_sound_obj

# Anything ffmpeg can decode will do, this is just to catch obvious mistakes
# (e.g. uploading a video or a text file) before running the conversion.
SUPPORTED_RECORDING_EXTENSIONS = ("wav", "mp3", "ogg", "m4a", "webm", "flac", "aac")
SUPPORTED_RECORDING_FORMATS_HELP = "Supported formats: " + ", ".join(
    ext.upper() for ext in SUPPORTED_RECORDING_EXTENSIONS)


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
            description='Collections:',
            disabled=False,
            layout=widgets.Layout(width='100%'),
            )
        self.collection_selector.observe(self._update_collection, 'value')

        self.sentence_selector = widgets.Dropdown(
            options=samples[self.collection].keys(),
            value=self.sentence,
            description='Sentences:',
            disabled=False,
            layout=widgets.Layout(width='100%'),
            )
        self.sentence_selector.observe(self._update_sentence, 'value')

        self.children = [self.collection_selector, self.sentence_selector]

    def selected_sample(self):
        # Empty collection ?
        if not self.sentence:
            return {}

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

    def set_selection(self, collection, sentence):
        self.collection = collection
        self.sentence = sentence
        self.value = self.selected_sample()


class UploadSample(widgets.VBox):
    value = Dict().tag(sync=True)

    def __init__(self, samples, my_samples_dir, **kwargs):
        super().__init__(**kwargs)

        self.samples = samples

        self.basepath = my_samples_dir
        pathlib.Path(self.basepath).mkdir(parents=True, exist_ok=True)

        self.w_file_upload = widgets.FileUpload(
            description='Upload WAV file',
            accept='wav',
            multiple=False,
        )
        self.w_sentence = widgets.Textarea(
            value='',
            placeholder='Type the sentence in kanji and kana',
            description='',
            disabled=False,
        )
        w_button = widgets.Button(description='Add sample', disabled=False)
        w_button.on_click(self._add_sample)

        self.w_notif = widgets.HTML(value='')
        self.w_file_upload.observe(self._clear_notif, 'value')
        self.w_sentence.observe(self._clear_notif, 'value')

        self.children = [
            widgets.Label(
                "Uploaded audio must be WAV format with 1 channel at 16kHz sampling frequency"),
            widgets.HBox([
                self.w_file_upload,
                self.w_sentence,
                widgets.VBox([w_button, self.w_notif]),
            ]),
        ]

        self.value = {}

    def _clear_notif(self, _):
        self.w_notif.value = ''

    def _add_sample(self, _):
        sentence = self.w_sentence.value
        if not sentence:
            return

        # Try parsing the sentence to make sure it is valid
        try:
            Sentence(sentence)
        except:
            import traceback
            self.w_notif.value = '<span style="color:#ff0000;">Invalid sentence !</span>'
            print(traceback.format_exc())
            return

        uploads = list(self.w_file_upload.value.items())
        # Double-checking with the _counter because sometimes the value is not cleared !
        if not uploads or self.w_file_upload._counter == 0:
            return

        filename, upload = uploads[0]

        filepath = os.path.join(self.basepath, filename)
        content = upload['content']

        try:
            check_wav_content_is_valid(content)
        except:
            import traceback
            self.w_notif.value = '<span style="color:#ff0000;">Invalid audio format !</span>'
            print(traceback.format_exc())
            return

        with open(filepath, 'wb') as output_file:
            output_file.write(content)

        self.value = {
            "filename": filepath,
            "sentence": sentence,
        }

        with self.w_file_upload.hold_sync(), self.w_sentence.hold_sync():
            self.w_file_upload.value.clear()
            self.w_file_upload._counter = 0
            self.w_sentence.value = ""
            self.w_notif.value = '<span style="color:#00ff00;">Successful !</span>'


class UploadRecording(widgets.VBox):
    """
    Lets the user submit a recording made outside of the app,
    as an alternative to recording directly in the browser.
    The uploaded audio is converted to the WAV format expected by SpeechRecord.
    """
    value = Dict().tag(sync=True)

    def __init__(self, my_recordings_dir, **kwargs):
        super().__init__(**kwargs)

        self.basepath = my_recordings_dir
        pathlib.Path(self.basepath).mkdir(parents=True, exist_ok=True)

        # Used to give each upload its own filename, so that uploading the same
        # file twice still notifies the observers.
        self.upload_count = 0

        self.w_file_upload = widgets.FileUpload(
            description='Upload',
            tooltip=f"Use a recording you made outside the app. "
                    f"{SUPPORTED_RECORDING_FORMATS_HELP}",
            icon='upload',
            accept=",".join(f".{ext}" for ext in SUPPORTED_RECORDING_EXTENSIONS),
            multiple=False,
        )
        # The label is rendered by the stylesheet, see the notebook
        self.w_file_upload.add_class('onsei-upload')
        self.w_file_upload.observe(self._add_recording, 'value')

        self.w_notif = widgets.HTML(value='')

        self.children = [self.w_file_upload, self.w_notif]

        self.value = {}

    def clear_notif(self):
        self.w_notif.value = ''

    def _add_recording(self, _):
        uploads = list(self.w_file_upload.value.items())
        # Double-checking with the _counter because sometimes the value is not cleared !
        if not uploads or self.w_file_upload._counter == 0:
            return

        filename, upload = uploads[0]

        # Clear right away: this frees the uploaded bytes and makes sure uploading
        # the same file again is picked up. Note that resetting the counter is what
        # rebuilds `value`, so `metadata` and `data` have to be emptied first.
        with self.w_file_upload.hold_sync():
            self.w_file_upload.metadata = []
            self.w_file_upload.data = []
            self.w_file_upload._counter = 0

        extension = os.path.splitext(filename)[1].lstrip('.').lower()
        if extension not in SUPPORTED_RECORDING_EXTENSIONS:
            self.w_notif.value = (
                '<span style="color:#ff0000;">Unsupported audio format !</span>'
            )
            return

        self.upload_count += 1
        # Keep the two paths distinct, ffmpeg refuses to convert a file onto itself
        original_filepath = os.path.join(
            self.basepath, f"upload_{self.upload_count}.{extension}")
        with open(original_filepath, 'wb') as output_file:
            output_file.write(upload['content'])

        wav_filepath = os.path.join(
            self.basepath, f"recording_{self.upload_count}.wav")
        try:
            convert_audio(original_filepath, wav_filepath)
            # Will raise if the conversion did not produce a usable recording
            parse_wav_file_to_sound_obj(wav_filepath)
        except Exception:
            logging.exception("Could not convert the uploaded recording")
            self.w_notif.value = (
                '<span style="color:#ff0000;">Could not read this audio file !</span>'
            )
            return

        self.w_notif.value = (
            f'<span style="color:#00ff00;">Using {html.escape(filename)}</span>'
        )
        self.value = {"filename": wav_filepath}
