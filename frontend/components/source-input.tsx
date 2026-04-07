'use client';

import { useState, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Youtube, Upload, Link2, FileVideo, X, Loader2, Play } from 'lucide-react';
import { isValidYouTubeUrl, extractYouTubeVideoId, getYouTubeThumbnail, formatFileSize } from '@/lib/helpers';
import { useI18n } from '@/lib/i18n';
import type { SourceType } from '@/lib/types';

interface SourceInputProps {
  onProcessUrl: (url: string) => void;
  onProcessFile: (file: File) => void;
  isProcessing: boolean;
  uploadProgress: number;
  sourceType: SourceType;
  onSourceTypeChange: (type: SourceType) => void;
}

export function SourceInput({ 
  onProcessUrl, 
  onProcessFile, 
  isProcessing, 
  uploadProgress,
  sourceType,
  onSourceTypeChange,
}: SourceInputProps) {
  const { t } = useI18n();
  const [url, setUrl] = useState('');
  const [urlError, setUrlError] = useState('');
  const [videoId, setVideoId] = useState<string | null>(null);
  const [thumbnailError, setThumbnailError] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateUrl = useCallback((value: string) => {
    if (!value.trim()) {
      setUrlError('');
      setVideoId(null);
      setThumbnailError(false);
      return false;
    }
    
    if (!isValidYouTubeUrl(value)) {
      setUrlError(t('invalidYoutubeUrl'));
      setVideoId(null);
      setThumbnailError(false);
      return false;
    }
    
    setUrlError('');
    const id = extractYouTubeVideoId(value);
    setVideoId(id);
    setThumbnailError(false);
    return true;
  }, [t]);

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setUrl(value);
    validateUrl(value);
  };

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      setUrl(text);
      validateUrl(text);
    } catch {
      // Clipboard access denied
    }
  };

  const handleSubmitUrl = () => {
    if (validateUrl(url) && !isProcessing) {
      onProcessUrl(url);
    }
  };

  const handleFileSelect = (file: File) => {
    const validTypes = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo'];
    if (!validTypes.includes(file.type)) {
      return;
    }
    setSelectedFile(file);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleSubmitFile = () => {
    if (selectedFile && !isProcessing) {
      onProcessFile(selectedFile);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Normalize URL for display
  const normalizeUrl = (rawUrl: string): string => {
    try {
      const urlObj = new URL(rawUrl);
      return urlObj.hostname + urlObj.pathname + urlObj.search;
    } catch {
      return rawUrl;
    }
  };

  return (
    <Card className="border-border/50 bg-card/80 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <Link2 className="w-4 h-4 text-primary" />
          {t('videoSource')}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={sourceType} onValueChange={(v) => onSourceTypeChange(v as SourceType)}>
          <TabsList className="grid grid-cols-2 mb-4 bg-secondary/50">
            <TabsTrigger value="youtube" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Youtube className="w-4 h-4" />
              {t('youtubeUrl')}
            </TabsTrigger>
            <TabsTrigger value="upload" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Upload className="w-4 h-4" />
              {t('uploadVideo')}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="youtube" className="space-y-4 mt-0">
            <div className="space-y-2">
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    placeholder="https://www.youtube.com/watch?v=..."
                    value={url}
                    onChange={handleUrlChange}
                    onKeyDown={(e) => e.key === 'Enter' && handleSubmitUrl()}
                    className="pr-20 bg-secondary/50 border-border/50 focus:border-primary/50"
                    disabled={isProcessing}
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    className="absolute right-1 top-1/2 -translate-y-1/2 h-7 text-xs text-muted-foreground hover:text-foreground"
                    onClick={handlePaste}
                    disabled={isProcessing}
                  >
                    {t('paste')}
                  </Button>
                </div>
              </div>
              {urlError && (
                <p className="text-xs text-destructive">{urlError}</p>
              )}
            </div>

            {/* YouTube Preview */}
            {videoId && (
              <div className="relative rounded-lg overflow-hidden border border-border/50 bg-secondary/30">
                <div className="aspect-video relative">
                  {!thumbnailError ? (
                    <img
                      src={getYouTubeThumbnail(videoId)}
                      alt={t('videoThumbnailAlt')}
                      className="w-full h-full object-cover"
                      onError={() => setThumbnailError(true)}
                    />
                  ) : (
                    /* Fallback preview when thumbnail fails */
                    <div className="w-full h-full bg-gradient-to-br from-secondary via-secondary/80 to-secondary/60 flex flex-col items-center justify-center">
                      <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center mb-3 border border-primary/30">
                        <Play className="w-8 h-8 text-primary ml-1" />
                      </div>
                      <span className="text-xs text-muted-foreground font-medium">{t('youtubeVideo')}</span>
                    </div>
                  )}
                  <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent" />
                  <div className="absolute bottom-3 left-3 right-3">
                    <p className="text-xs text-muted-foreground font-mono truncate">
                      {normalizeUrl(url)}
                    </p>
                  </div>
                </div>
              </div>
            )}

            <Button
              onClick={handleSubmitUrl}
              disabled={!url || !!urlError || isProcessing}
              className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
            >
              {isProcessing && sourceType === 'youtube' ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {t('extracting')}
                </>
              ) : (
                t('extractTabs')
              )}
            </Button>
          </TabsContent>

          <TabsContent value="upload" className="space-y-4 mt-0">
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              className="hidden"
            />

            {!selectedFile ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                className={`
                  border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all
                  ${isDragging 
                    ? 'border-primary bg-primary/5' 
                    : 'border-border/50 hover:border-primary/50 bg-secondary/30'
                  }
                `}
              >
                <FileVideo className="w-10 h-10 mx-auto mb-3 text-muted-foreground" />
                <p className="text-sm text-foreground mb-1">
                  {t('dropVideoHere')}
                </p>
                <p className="text-xs text-muted-foreground">
                  {t('orClickToSelect')}
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30 border border-border/50">
                  <FileVideo className="w-8 h-8 text-primary flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{selectedFile.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(selectedFile.size)}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-foreground"
                    onClick={clearFile}
                    disabled={isProcessing}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>

                {isProcessing && sourceType === 'upload' && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">{t('uploadProgress')}</span>
                      <span className="text-primary font-mono">{uploadProgress}%</span>
                    </div>
                    <Progress value={uploadProgress} className="h-1.5" />
                  </div>
                )}
              </div>
            )}

            <Button
              onClick={handleSubmitFile}
              disabled={!selectedFile || isProcessing}
              className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
            >
              {isProcessing && sourceType === 'upload' ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {uploadProgress < 100 ? t('uploading') : t('processing')}
                </>
              ) : (
                t('extractTabs')
              )}
            </Button>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
