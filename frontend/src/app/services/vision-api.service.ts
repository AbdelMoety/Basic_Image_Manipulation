import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import { CvOperation, ProcessResult } from '../models/operation.model';

@Injectable({ providedIn: 'root' })
export class VisionApiService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiBaseUrl;

  listOperations(): Observable<{ operations: CvOperation[] }> {
    return this.http.get<{ operations: CvOperation[] }>(`${this.base}/api/operations/`);
  }

  processImage(formData: FormData): Observable<ProcessResult> {
    return this.http.post<ProcessResult>(`${this.base}/api/process/`, formData);
  }

  mediaUrl(path: string | undefined): string | null {
    if (!path) {
      return null;
    }
    if (path.startsWith('http')) {
      return path;
    }
    return `${this.base}${path}`;
  }
}
