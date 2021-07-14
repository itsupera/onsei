# pip3 install fastapi python-multipart uvicorn
# uvicorn onsei.api:app --reload
"""
API to perform an audio comparison and get a graph
"""
import logging
import os
import subprocess
import traceback
from io import BytesIO
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Form, status, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from onsei.pyplot import plot_aligned_pitches_and_phonemes, plot_pitch_and_spectro, plot_pitch_and_phonemes
from onsei.speech_record import SpeechRecord, AlignmentError, NoPhonemeSegmentationError

app = FastAPI()


SUPPORTED_FILE_EXTENSIONS = {"wav", "mp3", "ogg"}


@app.post("/compare/graph.png")
def post_compare_graph_png(
    sentence: str = Form(...),
    show_all_graphs: bool = Form(False),
    teacher_audio_file: UploadFile = File(...),
    student_audio_file: UploadFile = File(...),
):
    for file, label in [(teacher_audio_file, "Reference audio"), (student_audio_file, "Your recording")]:
        extension = file.filename.split('.')[-1]
        # Check file extension first to make the error more user-friendly
        if extension not in SUPPORTED_FILE_EXTENSIONS:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                detail=f'{label} {file.filename} has unsupported extension {extension}, '
                                       f'should be one of the following: {",".join(SUPPORTED_FILE_EXTENSIONS)}')

    with TemporaryDirectory() as td:
        teacher_audio_filepath = os.path.join(td, teacher_audio_file.filename)
        with open(teacher_audio_filepath, "wb") as f:
            f.write(teacher_audio_file.file.read())
        student_audio_filepath = os.path.join(td, student_audio_file.filename)
        with open(student_audio_filepath, "wb") as f:
            f.write(student_audio_file.file.read())

        logging.debug(f"Converting {teacher_audio_filepath} and {student_audio_filepath} to WAV 16KHz mono")

        teacher_wav_filepath = os.path.join(td, "teacher.wav")
        convert_audio(teacher_audio_filepath, teacher_wav_filepath)
        student_wav_filepath = os.path.join(td, "student.wav")
        convert_audio(student_audio_filepath, student_wav_filepath)

        logging.debug(f"Comparing {teacher_wav_filepath} with {student_wav_filepath}")

        try:
            teacher_rec = SpeechRecord(teacher_wav_filepath, sentence, name="Teacher")
            student_rec = SpeechRecord(student_wav_filepath, sentence, name="Student")
            student_rec.align_with(teacher_rec)
            mean_distance = student_rec.compare_pitch()
        except AlignmentError:
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f'Could not align your speech with the reference, try recording again')
        except NoPhonemeSegmentationError as exc:
            logging.error(traceback.format_exc())
            if exc.record == student_rec:
                detail = f'Could not segment the phonemes in your speech, try recording again'
            else:
                detail = f'Could not segment the phonemes in the reference audio'
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=detail)
        except Exception:
            logging.error(traceback.format_exc())
            "No warping path found compatible with the local constraints"
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f'Something went wrong on the server, not your fault :(')

        # Transform the distance into a score from 0 to 100
        score = int(1.0 / (mean_distance + 1.0) * 100)

    if show_all_graphs:
        plt.figure(figsize=(12, 6))
        plt.subplot(311)
        plt.title(f"Similarity score: {score}%")
        plot_aligned_pitches_and_phonemes(student_rec)
        plt.subplot(312)
        plot_pitch_and_phonemes(student_rec, 'r', "Your recording")
        plt.subplot(313)
        plot_pitch_and_phonemes(teacher_rec, 'b', "Reference audio")
    else:
        plt.figure(figsize=(12, 4))
        plt.title(f"Similarity score: {score}%")
        plot_aligned_pitches_and_phonemes(student_rec)

    b = BytesIO()
    plt.savefig(b, format='png')
    b.seek(0)

    return StreamingResponse(b, media_type="image/png")


@app.get("/")
async def get_root():
    """ Form for testing """
    content = """
<body>
<form action="/compare/graph.png" enctype="multipart/form-data" method="post">
Teacher audio file: <input name="teacher_audio_file" type="file"></br>
Student audio file: <input name="student_audio_file" type="file"></br>
Sentence: <input name="sentence" type="text"></br>
<input name="show_all_graphs" type="checkbox">Show all graphs ?</br>
<input type="submit">
</form>
</br>
<a href="https://github.com/itsupera/onsei#readme">What is this ?</a></br>
</body>
    """
    return HTMLResponse(content=content)


def convert_audio(original_audio_filepath: str, converted_wav_filepath: str) -> None:
    p = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", original_audio_filepath, "-ar", "16000", "-ac", "1",
         converted_wav_filepath])
    p.wait()