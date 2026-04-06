import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')

templates = Jinja2Templates(directory='templates')

@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/newsletter', response_class=HTMLResponse)
async def read_newsletter(request: Request):
    return templates.TemplateResponse('newsletter.html', {'request': request})

@app.post('/submit_newsletter')
async def submit_newsletter(request: Request):
    form_data = await request.form()
    # Process the form data (e.g., save to a database)
    # For now, just return a success message
    return {'message': 'Newsletter submitted successfully!'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
