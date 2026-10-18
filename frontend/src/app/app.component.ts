import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnInit, ViewChild, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { CvOperation, ProcessParams, ProcessResult } from './models/operation.model';
import { VisionApiService } from './services/vision-api.service';

const AUTO_GRAY_OPS = new Set([
  'rgb2gray',
  'threshold',
  'iter_threshold',
  'local_threshold',
  'equalize',
  'auto_canny',
]);

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent implements OnInit {
  private readonly api = inject(VisionApiService);

  @ViewChild('primaryInput') private primaryInput?: ElementRef<HTMLInputElement>;

  title = 'Computer Vision Workbench';
  operations: CvOperation[] = [];
  groupedOperations: { group: string; items: CvOperation[] }[] = [];

  selectedOperation = 'sobel';
  primaryFile: File | null = null;
  secondaryFile: File | null = null;
  primaryPreview: string | null = null;
  secondaryPreview: string | null = null;

  params: ProcessParams = {
    filter_type: 'sobel',
    convert_gray: false,
    kernel_size: 3,
    sigma: 1.0,
    canny_sigma: 0.33,
    thresh_val: 128,
    iter_diff: 0.001,
    local_win: 15,
    local_c: 5.0,
    noise_mean: 0,
    noise_sigma: 25,
    sp_amount: 0.05,
    sp_salt_ratio: 0.5,
    uniform_low: -25,
    uniform_high: 25,
    cutoff_low: 25,
    cutoff_high: 25,
    alpha: 0.5,
    beta: 0.5,
  };

  loading = false;
  error: string | null = null;
  info: string | null = null;
  result: ProcessResult | null = null;

  /** Operation that produced the current result (the dropdown may have changed since). */
  resultOperation: string | null = null;
  /** How many times "Further" has been used on the current image (0 = untouched upload). */
  chainStep = 0;
  furthering = false;

  ngOnInit(): void {
    this.api.listOperations().subscribe({
      next: (res) => {
        this.operations = res.operations;
        this.groupedOperations = this.groupByCategory(res.operations);
      },
      error: () => {
        this.error = 'Backend not reachable — run: python manage.py runserver';
      },
    });
  }

  get op(): string {
    return this.selectedOperation;
  }

  get showGrayToggle(): boolean {
    return !AUTO_GRAY_OPS.has(this.op);
  }

  get showImage2(): boolean {
    return this.op === 'hybrid';
  }

  get showKernel(): boolean {
    return ['mean', 'gaussian', 'median', 'auto_canny'].includes(this.op);
  }

  get showSigma(): boolean {
    return this.op === 'gaussian';
  }

  get showCannySigma(): boolean {
    return this.op === 'auto_canny';
  }

  get showThresh(): boolean {
    return this.op === 'threshold';
  }

  get showIterDiff(): boolean {
    return this.op === 'iter_threshold';
  }

  get showLocal(): boolean {
    return this.op === 'local_threshold';
  }

  get showNoise(): boolean {
    return this.op === 'awgn';
  }

  get showSaltPepper(): boolean {
    return this.op === 'salt_pepper';
  }

  get showUniform(): boolean {
    return this.op === 'uniform_noise';
  }

  get showCutoffLow(): boolean {
    return this.op === 'lowpass' || this.op === 'hybrid';
  }

  get showCutoffHigh(): boolean {
    return this.op === 'highpass' || this.op === 'hybrid';
  }

  get showHybridWeights(): boolean {
    return this.op === 'hybrid';
  }

  /** The histogram operation outputs a chart, not an image that can be processed again. */
  get canFurther(): boolean {
    return (
      !!this.result?.processed_image &&
      this.resultOperation !== 'plot_histograms' &&
      !this.loading &&
      !this.furthering
    );
  }

  /**
   * Promote the current output to be the input image so another operation can be
   * applied to it. The Original panel then shows the previous output.
   */
  async further(): Promise<void> {
    const outUrl = this.mediaUrl(this.result?.processed_image);
    if (!this.result || !outUrl || !this.canFurther) {
      return;
    }

    this.furthering = true;
    this.error = null;
    try {
      const resp = await fetch(outUrl);
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const blob = await resp.blob();

      // Short client-side name: the backend prefixes names on every pass, so reusing
      // the server filename would grow until it hits the filesystem limit.
      const ext = (outUrl.split('?')[0].match(/\.(png|jpe?g|bmp|webp|tiff?)$/i)?.[1] ?? 'png').toLowerCase();
      this.chainStep += 1;
      const file = new File([blob], `further_${this.chainStep}.${ext}`, {
        type: blob.type || `image/${ext === 'jpg' ? 'jpeg' : ext}`,
      });

      this.primaryFile = file;
      this.revokePreview(this.primaryPreview);
      this.primaryPreview = URL.createObjectURL(file);
      this.syncPrimaryInput(file);

      // Original panel -> previous output; Output panel waits for the next run.
      this.result = {
        uploaded_image: this.result.processed_image,
        uploaded_image_2: this.result.uploaded_image_2,
      };
      this.resultOperation = null;
      this.info = 'Output is now the input image. Choose an operation and press Process image.';
    } catch {
      this.error = 'Could not load the output image to continue from it.';
    } finally {
      this.furthering = false;
    }
  }

  onOperationChange(): void {
    this.params.filter_type = this.selectedOperation;
    this.error = null;
    this.info = null;
  }

  onPrimarySelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.primaryFile = file;
    this.chainStep = 0;
    this.revokePreview(this.primaryPreview);
    this.primaryPreview = file ? URL.createObjectURL(file) : null;
  }

  onSecondarySelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.secondaryFile = file;
    this.revokePreview(this.secondaryPreview);
    this.secondaryPreview = file ? URL.createObjectURL(file) : null;
  }

  process(): void {
    if (!this.primaryFile) {
      this.error = 'Upload a primary image first.';
      return;
    }
    if (this.op === 'hybrid' && !this.secondaryFile) {
      this.error = 'Hybrid mode needs a second image.';
      return;
    }

    const form = new FormData();
    form.append('image', this.primaryFile);
    if (this.secondaryFile) {
      form.append('image_2', this.secondaryFile);
    }
    form.append('filter_type', this.selectedOperation);
    if (this.params.convert_gray) {
      form.append('convert_gray', 'on');
    }

    const numericKeys: (keyof ProcessParams)[] = [
      'kernel_size',
      'sigma',
      'canny_sigma',
      'thresh_val',
      'iter_diff',
      'local_win',
      'local_c',
      'noise_mean',
      'noise_sigma',
      'sp_amount',
      'sp_salt_ratio',
      'uniform_low',
      'uniform_high',
      'cutoff_low',
      'cutoff_high',
      'alpha',
      'beta',
    ];
    for (const key of numericKeys) {
      form.append(key, String(this.params[key]));
    }

    this.loading = true;
    this.error = null;
    this.info = null;
    this.result = null;

    const operationUsed = this.selectedOperation;

    this.api.processImage(form).subscribe({
      next: (res) => {
        this.loading = false;
        if (res.error) {
          this.error = res.error;
          return;
        }
        this.result = res;
        this.resultOperation = operationUsed;
        this.info = res.info ?? null;
      },
      error: (err) => {
        this.loading = false;
        this.error = err.error?.error ?? 'Processing failed. Check the backend logs.';
      },
    });
  }

  mediaUrl(path: string | undefined): string | null {
    return this.api.mediaUrl(path);
  }

  private groupByCategory(ops: CvOperation[]): { group: string; items: CvOperation[] }[] {
    const map = new Map<string, CvOperation[]>();
    for (const op of ops) {
      const list = map.get(op.group) ?? [];
      list.push(op);
      map.set(op.group, list);
    }
    return Array.from(map.entries()).map(([group, items]) => ({ group, items }));
  }

  /** Make the "Image 1" file input display the promoted file instead of the old upload. */
  private syncPrimaryInput(file: File): void {
    const input = this.primaryInput?.nativeElement;
    if (!input) {
      return;
    }
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    } catch {
      /* Older browsers: the preview and primaryFile are still correct. */
    }
  }

  private revokePreview(url: string | null): void {
    if (url) {
      URL.revokeObjectURL(url);
    }
  }
}
