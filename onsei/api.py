# pip3 install fastapi python-multipart uvicorn
# uvicorn onsei.api:app --reload
"""
API to perform an audio comparison and get a graph
"""

import os
from io import BytesIO
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Form, status, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from onsei.pyplot import plot_aligned_pitches_and_phonemes
from onsei.speech_record import SpeechRecord

app = FastAPI()


SUPPORTED_FILE_EXTENSIONS = {"wav"}
SUPPORTED_CONTENT_TYPES = {"audio/vnd.wave", "audio/wav", "audio/wave", "audio/x-wav"}


@app.post("/compare/graph.png")
def post_compare_graph_png(
    sentence: str = Form(...),
    teacher_wav_file: UploadFile = File(...),
    student_wav_file: UploadFile = File(...),
):
    for file, name in [(teacher_wav_file, ""), (student_wav_file, "Your recording")]:
        extension = file.filename.split('.')[-1]
        # Check file extension first to make the error more user-friendly
        if extension not in SUPPORTED_FILE_EXTENSIONS:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                detail=f'{name} {file.filename} has unsupported extension {extension}, '
                                       f'should be one of the following: {",".join(SUPPORTED_FILE_EXTENSIONS)}')
        if file.content_type not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                detail=f'{name} {file.filename} has unsupported format {file.content_type}, '
                                       f'should be one of the following: {",".join(SUPPORTED_CONTENT_TYPES)}')

    with TemporaryDirectory() as td:
        teacher_wav_filename = os.path.join(td, teacher_wav_file.filename)
        student_wav_filename = os.path.join(td, student_wav_file.filename)
        with open(teacher_wav_filename, 'wb') as fd:
            content = teacher_wav_file.file.read()
            fd.write(content)
        with open(student_wav_filename, 'wb') as fd:
            content = student_wav_file.file.read()
            fd.write(content)

        print(f"Comparing {teacher_wav_filename} with {student_wav_filename}")

        try:
            teacher_rec = SpeechRecord(teacher_wav_filename, sentence, name="Teacher")
            student_rec = SpeechRecord(student_wav_filename, sentence, name="Student")
            student_rec.align_with(teacher_rec)
            mean_distance = student_rec.compare_pitch()
        except Exception:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f'Something went wrong on the server, not your fault :(')

        # Transform the distance into a score from 0 to 100
        score = int(1.0 / (mean_distance + 1.0) * 100)

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
Teacher audio file: <input name="teacher_wav_file" type="file"></br>
Student audio file: <input name="student_wav_file" type="file"></br>
Sentence: <input name="sentence" type="text"></br>
<input type="submit">
</form>
</br>
<a href="https://github.com/itsupera/onsei#readme">What is this ?</a></br>
</body>
    """
    return HTMLResponse(content=content)
