import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';

import { AppComponent } from './app.component';

const ID = 'a'.repeat(32);
const tick = () => new Promise<void>((r) => setTimeout(r));

// Runner-agnostic on purpose (no runner-specific mocking APIs): works with Karma+Jasmine, Jest and Vitest.
type UrlApi = { createObjectURL?: unknown; revokeObjectURL?: unknown };

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('loads operations on init', () => {
    const httpMock = TestBed.inject(HttpTestingController);
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();

    const req = httpMock.expectOne('/api/operations/');
    expect(req.request.method).toBe('GET');
    req.flush({ operations: [] });
    httpMock.verify();
  });

  describe('upload once, process many', () => {
    let httpMock: HttpTestingController;
    let app: AppComponent;

    afterEach(() => {
      urlApi.createObjectURL = original.create;
      urlApi.revokeObjectURL = original.revoke;
      console.warn = originalWarn;
    });

    const pick = (name = 'photo.png') => {
      const file = new File([new Uint8Array([1, 2, 3])], name, { type: 'image/png' });
      app.onPrimarySelected({ target: { files: [file] } } as unknown as Event);
      return file;
    };

    const urlApi = URL as unknown as UrlApi;
    const original = { create: urlApi.createObjectURL, revoke: urlApi.revokeObjectURL };
    const originalWarn = console.warn;

    beforeEach(() => {
      urlApi.createObjectURL = () => 'blob:preview'; // missing in some test DOMs
      urlApi.revokeObjectURL = () => undefined;
      httpMock = TestBed.inject(HttpTestingController);
      const fixture = TestBed.createComponent(AppComponent);
      app = fixture.componentInstance;
      fixture.detectChanges();
      httpMock.expectOne('/api/operations/').flush({ operations: [] });
    });

    it('uploads on selection, then sends only image_id with every operation', async () => {
      pick();
      httpMock.expectOne('/api/upload/').flush({ image_id: ID, url: '/media/src.png' });

      for (const op of ['sobel', 'manual_canny', 'gaussian']) {
        app.selectedOperation = op;
        app.onOperationChange();
        const done = app.process();
        await tick();
        const req = httpMock.expectOne('/api/process/');
        const body = req.request.body as FormData;
        expect(body.get('image_id')).toBe(ID);
        expect(body.get('image')).toBeNull();
        expect(body.get('filter_type')).toBe(op);
        req.flush({ processed_image: '/media/out.png', info: 'ok' });
        await done;
      }
      httpMock.expectNone('/api/upload/'); // never uploaded again
      httpMock.verify();
    });

    it('sends the detections list for Manual Canny + Hough shapes', async () => {
      pick();
      httpMock.expectOne('/api/upload/').flush({ image_id: ID, url: '/media/src.png' });
      app.selectedOperation = 'manual_canny';
      app.onOperationChange();
      app.detectLines = true;
      app.detectEllipses = true;
      const done = app.process();
      await tick();
      const req = httpMock.expectOne('/api/process/');
      expect((req.request.body as FormData).get('detections')).toBe('lines,ellipses');
      req.flush({ processed_image: '/media/out.png' });
      await done;
    });

    it('waits for an upload that is still running instead of uploading twice', async () => {
      pick();
      const upload = httpMock.expectOne('/api/upload/'); // not flushed yet
      const done = app.process();
      await tick();
      httpMock.expectNone('/api/process/');               // held back until the upload finishes
      upload.flush({ image_id: ID, url: '/media/src.png' });
      await tick();
      const req = httpMock.expectOne('/api/process/');
      expect((req.request.body as FormData).get('image_id')).toBe(ID);
      req.flush({ processed_image: '/media/out.png' });
      await done;
    });

    it('falls back to sending the file if the upload endpoint fails', async () => {
      console.warn = () => undefined; // the component logs a warning when an upload fails
      const file = pick();
      httpMock.expectOne('/api/upload/').flush('nope', { status: 404, statusText: 'Not Found' });
      const done = app.process();
      await tick();
      const req = httpMock.expectOne('/api/process/');
      const body = req.request.body as FormData;
      expect(body.get('image_id')).toBeNull();
      expect((body.get('image') as File).name).toBe(file.name);
      req.flush({ processed_image: '/media/out.png' });
      await done;
    });

    it('uploads again when a different file is picked', async () => {
      pick('one.png');
      httpMock.expectOne('/api/upload/').flush({ image_id: ID, url: '/media/1.png' });
      pick('two.png');
      httpMock.expectOne('/api/upload/').flush({ image_id: 'b'.repeat(32), url: '/media/2.png' });
      const done = app.process();
      await tick();
      const req = httpMock.expectOne('/api/process/');
      expect((req.request.body as FormData).get('image_id')).toBe('b'.repeat(32));
      req.flush({ processed_image: '/media/out.png' });
      await done;
    });
  });
});