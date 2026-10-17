import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
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
    cutoff_low: 25,
    cutoff_high: 25,
    alpha: 0.5,
    beta: 0.5,
  };

  loading = false;
  error: string | null = null;
  info: string | null = null;
  result: ProcessResult | null = null;

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

  get showCutoffLow(): boolean {
    return this.op === 'lowpass' || this.op === 'hybrid';
  }

  get showCutoffHigh(): boolean {
    return this.op === 'highpass' || this.op === 'hybrid';
  }

  get showHybridWeights(): boolean {
    return this.op === 'hybrid';
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

    this.api.processImage(form).subscribe({
      next: (res) => {
        this.loading = false;
        if (res.error) {
          this.error = res.error;
          return;
        }
        this.result = res;
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

  private revokePreview(url: string | null): void {
    if (url) {
      URL.revokeObjectURL(url);
    }
  }
}
