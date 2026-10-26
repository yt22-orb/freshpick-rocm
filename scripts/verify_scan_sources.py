"""Verify separate Camera/Gallery panels, uploads, drop handling and media lifecycle."""
import base64,json
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

OUT=Path('runs/scan-source-verification');OUT.mkdir(exist_ok=True,parents=True)
photo=Path('runs/evaluation-web/images/banana_direct_19231740.jpg').resolve()
encoded=base64.b64encode(photo.read_bytes()).decode()
checks=[];errors=[]
with sync_playwright() as tool:
 browser=tool.chromium.launch(headless=True,args=['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream'])
 context=browser.new_context(viewport={'width':390,'height':844},permissions=['camera'])
 page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
 page.goto('http://localhost:8000')
 page.locator('#home-screen [data-action="scan"]').click()
 expect(page.locator('#camera-frame')).to_have_class('camera-frame live')
 expect(page.locator('[data-action="capture"]')).to_be_enabled()
 page.screenshot(path=str(OUT/'camera.png'),full_page=True)
 # Retain the real browser track to prove that Gallery switches stop it.
 page.evaluate('window.previousTrack = cameraStream.getVideoTracks()[0]')
 choosers=[];page.on('filechooser',lambda chooser:choosers.append(chooser))
 page.locator('#source-gallery').click()
 expect(page.locator('#gallery-panel')).to_be_visible()
 expect(page.locator('#camera-panel')).not_to_be_visible()
 for action in ['capture','switch-camera','flash','start-camera']:
  expect(page.locator(f'[data-action="{action}"]')).not_to_be_visible()
 assert page.evaluate('window.previousTrack.readyState')=='ended'
 assert page.evaluate('cameraStream === null && cameraVideo.srcObject === null')
 assert not choosers
 page.screenshot(path=str(OUT/'gallery.png'),full_page=True)
 checks.append('Gallery tab shows the drop area, opens no file picker, hides all camera controls and stops camera tracks')
 # File picker is an explicit secondary action, and uses the actual model service.
 with page.expect_file_chooser() as chooser:
  page.locator('[data-action="choose-gallery-file"]').click()
 with page.expect_response('**/predict',timeout=90000) as response:
  chooser.value.set_files(str(photo))
 assert response.value.status==200
 payload=response.value.json();assert payload['produce']=='banana'
 expect(page.locator('#report-screen')).to_be_visible()
 checks.append('Choose image opens the native picker; the selected original image reaches real inference and Report')
 page.locator('#report-screen [data-action="scan"]').click()
 expect(page.locator('#gallery-panel')).to_be_visible()
 # Construct native browser File/DataTransfer objects, not direct calls to upload helpers.
 def transfer(files):
  return page.evaluate_handle('''files => { const dt = new DataTransfer();
   for (const item of files) { const bytes = item.base64 ? Uint8Array.from(atob(item.base64), c=>c.charCodeAt(0)) : new Uint8Array(item.size || 10);
    dt.items.add(new File([bytes], item.name, {type:item.type})); }
   return dt; }''',files)
 dt=transfer([{'name':'banana.jpg','type':'image/jpeg','base64':encoded}])
 page.locator('#gallery-dropzone').dispatch_event('dragenter',{'dataTransfer':dt})
 expect(page.locator('#gallery-dropzone')).to_have_class('gallery-dropzone is-dragging')
 with page.expect_response('**/predict',timeout=90000) as response:
  page.locator('#gallery-dropzone').dispatch_event('drop',{'dataTransfer':dt})
 assert response.value.status==200 and response.value.json()['produce']=='banana'
 expect(page.locator('#report-screen')).to_be_visible()
 checks.append('Dragging highlights the drop area and dropping an image performs real inference')
 page.locator('#report-screen [data-action="scan"]').click()
 requests=[];page.on('request',lambda req:requests.append(req.url) if req.url.endswith('/predict') else None)
 for files,message in [
  ([{'name':'notes.txt','type':'text/plain'}],'Choose a JPG'),
  ([{'name':'large.jpg','type':'image/jpeg','size':20*1024*1024+1}],'larger than 20 MB'),
  ([{'name':'one.jpg','type':'image/jpeg'},{'name':'two.jpg','type':'image/jpeg'}],'one image at a time'),
 ]:
  data=transfer(files);page.locator('#gallery-dropzone').dispatch_event('drop',{'dataTransfer':data})
  expect(page.locator('#scan-notice')).to_contain_text(message)
  expect(page.locator('#gallery-panel')).to_be_visible()
 assert not requests
 checks.append('Wrong file types, oversize files and multiple files are rejected without inference or camera controls')
 with page.expect_response('**/predict') as response:
  page.locator('#gallery-input').set_input_files({'name':'invalid.jpg','mimeType':'image/jpeg','buffer':b'bad bytes'})
 assert response.value.status==400
 expect(page.locator('#scan-notice')).to_contain_text('Invalid or unreadable image')
 expect(page.locator('#gallery-panel')).to_be_visible()
 expect(page.locator('[data-action="capture"]')).not_to_be_visible()
 checks.append('Invalid-image errors return to Gallery with the upload controls and no shutter')
 # Tab keyboard semantics; Gallery itself supports Enter to open the file picker.
 page.locator('#source-gallery').focus();page.keyboard.press('ArrowLeft')
 expect(page.locator('#camera-panel')).to_be_visible()
 expect(page.locator('#camera-frame')).to_have_class('camera-frame live')
 page.locator('#source-camera').focus();page.keyboard.press('ArrowRight')
 expect(page.locator('#gallery-panel')).to_be_visible()
 page.locator('#gallery-dropzone').focus()
 with page.expect_file_chooser():page.keyboard.press('Enter')
 checks.append('Keyboard tab switching and Enter-to-choose work without exposing camera controls in Gallery')
 for width in [320,390,768,1440]:
  page.set_viewport_size({'width':width,'height':900})
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),width
 checks.append('Gallery fits 320/390/768/1440px widths')
 # A delayed camera permission result must be closed after switching to Gallery.
 page.evaluate('''() => { window.savedGetUserMedia=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>window.resolvePendingCamera=resolve); }''')
 page.locator('#source-camera').click()
 expect(page.locator('#camera-start')).to_be_disabled()
 page.locator('#source-gallery').click()
 page.evaluate('''async () => { const stream=await window.savedGetUserMedia({video:true,audio:false});
  window.lateTrack=stream.getVideoTracks()[0]; window.resolvePendingCamera(stream); }''')
 page.wait_for_function('window.lateTrack.readyState === "ended"')
 expect(page.locator('#gallery-panel')).to_be_visible()
 checks.append('Late camera permission results close their tracks instead of restarting the camera in Gallery')
 # Explicit errors and a successful retry, with real browser capture afterwards.
 for error,message in [('NotAllowedError','Camera permission was blocked'),('NotFoundError','No camera was found'),('NotReadableError','camera is busy')]:
  page.evaluate('(name)=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException("Camera test",name);};}',error)
  page.locator('#source-camera').click()
  expect(page.locator('#scan-notice')).to_contain_text(message)
  expect(page.locator('#camera-start')).to_be_enabled()
  expect(page.locator('[data-action="capture"]')).to_be_disabled()
  page.locator('#source-gallery').click()
 page.evaluate('() => { navigator.mediaDevices.getUserMedia = window.savedGetUserMedia; }')
 page.locator('#source-camera').click()
 expect(page.locator('[data-action="capture"]')).to_be_enabled()
 with page.expect_response('**/predict',timeout=90000) as response:
  page.locator('[data-action="capture"]').click()
 assert response.value.status==200
 expect(page.locator('#report-screen')).to_be_visible()
 checks.append('Permission, missing-device and busy-camera states offer retry; simulated camera capture then reaches real inference')
 assert not errors,errors
 browser.close()
(OUT/'checks.json').write_text(json.dumps({'checks':checks,'javascript_errors':errors,'physical_camera':'Not exposed to this runtime; simulated browser camera used.'},indent=2))
print(json.dumps(checks,indent=2))
