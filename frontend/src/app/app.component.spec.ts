import { ChangeDetectorRef } from '@angular/core';
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

  it('shows the SIFT operations as their own section of the Operation dropdown', () => {
    const httpMock = TestBed.inject(HttpTestingController);
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    httpMock.expectOne('/api/operations/').flush({
      operations: [
        { id: 'sobel', label: 'Sobel Edge Detection', group: 'Edge Detection' },
        { id: 'sift_keypoints', label: 'SIFT Keypoints (Scale & Orientation)', group: 'Feature Matching (SIFT)' },
        { id: 'sift_match_ssd', label: 'SIFT Feature Matching (SSD)', group: 'Feature Matching (SIFT)' },
      ],
    });
    // The list arrives in an HTTP callback; mark the view dirty so this also passes without zone.js.
    fixture.componentRef.injector.get(ChangeDetectorRef).markForCheck();
    fixture.detectChanges();

    const groups = Array.from(fixture.nativeElement.querySelectorAll('optgroup')) as HTMLOptGroupElement[];
    expect(groups.map((g) => g.label)).toEqual(['Edge Detection', 'Feature Matching (SIFT)']);
    expect(Array.from(groups[1].querySelectorAll('option')).map((o) => (o as HTMLOptionElement).value)).toEqual([
      'sift_keypoints',
      'sift_match_ssd',
    ]);
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

    describe('SIFT feature matching section', () => {
      const pickSecond = (name = 'second.png') => {
        const file = new File([new Uint8Array([4, 5, 6])], name, { type: 'image/png' });
        app.onSecondarySelected({ target: { files: [file] } } as unknown as Event);
        return file;
      };
      const ID2 = 'b'.repeat(32);

      it('asks for Image 2 only for the matching operations', () => {
        for (const [op, needs] of [
          ['sift_match_ssd', true],
          ['sift_match_ncc', true],
          ['sift_keypoints', false],
          ['sift_scale_space', false],
          ['sobel', false],
        ] as const) {
          app.selectedOperation = op;
          app.onOperationChange();
          expect(app.showImage2).toBe(needs);
        }
        app.selectedOperation = 'sift_match_ssd';
        expect(app.image2Label).toBe('Image 2 (to match)');
        app.selectedOperation = 'hybrid';
        expect(app.image2Label).toBe('Image 2 (hybrid)');
      });

      it('hides the manual grayscale toggle (SIFT converts on the server)', () => {
        app.selectedOperation = 'sift_keypoints';
        expect(app.showGrayToggle).toBe(false);
      });

      it('does not call the backend when matching without a second image', async () => {
        pick();
        httpMock.expectOne('/api/upload/').flush({ image_id: ID, url: '/media/src.png' });
        app.selectedOperation = 'sift_match_ncc';
        app.onOperationChange();
        await app.process();
        expect(app.error).toContain('second image');
        httpMock.expectNone('/api/process/');
      });

      it('sends both image ids and the SIFT settings for matching', async () => {
        pick();
        httpMock.expectOne('/api/upload/').flush({ image_id: ID, url: '/media/src.png' });
        pickSecond();
        httpMock.expectOne('/api/upload/').flush({ image_id: ID2, url: '/media/second.png' });
        app.selectedOperation = 'sift_match_ssd';
        app.onOperationChange();
        app.params.sift_ratio = 0.7;
        app.params.sift_draw = 25;
        const done = app.process();
        await tick();
        const req = httpMock.expectOne('/api/process/');
        const body = req.request.body as FormData;
        expect(body.get('filter_type')).toBe('sift_match_ssd');
        expect(body.get('image_id')).toBe(ID);
        expect(body.get('image_2_id')).toBe(ID2);
        expect(body.get('sift_ratio')).toBe('0.7');
        expect(body.get('sift_draw')).toBe('25');
        expect(body.get('sift_contrast')).toBe('0.01');
        expect(body.get('sift_max_size')).toBe('400');
        req.flush({ processed_image: '/media/out.png', info: 'ok' });
        await done;
      });

      it('treats SIFT results as figures that cannot be fed back in as the input', async () => {
        pick();
        httpMock.expectOne('/api/upload/').flush({ image_id: ID, url: '/media/src.png' });
        app.selectedOperation = 'sift_keypoints';
        app.onOperationChange();
        const done = app.process();
        await tick();
        httpMock.expectOne('/api/process/').flush({ processed_image: '/media/out.png' });
        await done;
        expect(app.runs[0].isChart).toBe(true);
        expect(app.canFurther(app.runs[0])).toBe(false);

        // ...while an ordinary operation's result still can be
        app.selectedOperation = 'sobel';
        app.onOperationChange();
        const done2 = app.process();
        await tick();
        httpMock.expectOne('/api/process/').flush({ processed_image: '/media/out2.png' });
        await done2;
        expect(app.runs[0].isChart).toBe(false);
        expect(app.canFurther(app.runs[0])).toBe(true);
      });
    });
  });
});