import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

/**
 * Base URL of the Django API. Keep it identical to the one VisionApiService uses
 * (the unit test expects '/api/operations/', so '/api' matches your setup).
 */
export const API_BASE = '/api';

export interface UploadResponse {
  image_id: string;
  url: string;
}

/** Uploads an image once; later /process/ calls send the returned image_id instead of the file. */
@Injectable({ providedIn: 'root' })
export class ImageUploadService {
  private readonly http = inject(HttpClient);

  upload(file: File): Observable<UploadResponse> {
    const form = new FormData();
    form.append('image', file);
    return this.http.post<UploadResponse>(`${API_BASE}/upload/`, form);
  }
}
