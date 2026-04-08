'use client';

import { useState, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { 
  Upload, Link2, FileVideo, X, Loader2, Play, 
  ChevronDown, AlertTriangle, Shield, ExternalLink,
  Download, Monitor, Cookie, Info
} from 'lucide-react';
import { isValidVideoUrl, extractYouTubeVideoId, getYouTubeThumbnail, formatFileSize, getVideoPlatformLabel } from '@/lib/helpers';
import { useI18n } from '@/lib/i18n';
import type { SourceType } from '@/lib/types';

interface SourceInputProps {
  onProcessUrl: (url: string, cookiesText?: string) => void;
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
  
  // Advanced cookies state (future-ready)
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [useCookies, setUseCookies] = useState(false);
  const [cookiesText, setCookiesText] = useState('');

  const validateUrl = useCallback((value: string) => {
    if (!value.trim()) {
      setUrlError('');
      setVideoId(null);
      setThumbnailError(false);
      return false;
    }
    
    const id = extractYouTubeVideoId(value);
    if (!isValidVideoUrl(value)) {
      setUrlError(t('invalidVideoUrl'));
      setVideoId(id);
      setThumbnailError(false);
      return false;
    }

    setUrlError('');
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
      // Future-ready: pass cookies_text if enabled
      onProcessUrl(url, useCookies && cookiesText.trim() ? cookiesText.trim() : undefined);
    }
  };

  const handleClearCookies = () => {
    setCookiesText('');
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
          {/* Upload is now first and default */}
          <TabsList className="grid grid-cols-2 mb-4 bg-secondary/50">
            <TabsTrigger value="upload" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Upload className="w-4 h-4" />
              {t('uploadVideo')}
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary font-medium ml-1">
                {t('uploadRecommended')}
              </span>
            </TabsTrigger>
            <TabsTrigger value="youtube" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Link2 className="w-4 h-4" />
              {t('youtubeUrl')}
            </TabsTrigger>
          </TabsList>

          {/* Upload Tab - Now First */}
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

            {/* Downie 4 Helper Card */}
            <div className="rounded-lg border border-border/30 bg-secondary/20 p-4 mt-4">
              <div className="flex items-start gap-3">
                <Download className="w-5 h-5 text-muted-foreground mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground mb-1">
                    {t('needToDownload')}
                  </p>
                  <p className="text-xs text-muted-foreground mb-3">
                    {t('downloadHelperDesc')}
                  </p>
                  <a
                    href="https://software.charliemonroe.net/downie.php"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs text-primary hover:underline"
                  >
                    {t('tryDownie')}
                    <ExternalLink className="w-3 h-3" />
                  </a>
                  <span className="text-xs text-muted-foreground ml-2">
                    ({t('downieMacOnly')})
                  </span>
                </div>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-muted-foreground font-medium">
                  {t('externalTool')}
                </span>
              </div>
            </div>
          </TabsContent>

          {/* Video URL Tab (formerly YouTube) */}
          <TabsContent value="youtube" className="space-y-4 mt-0">
            <div className="space-y-2">
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    placeholder="https://www.youtube.com/watch?v=... or https://www.bilibili.com/..."
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

            {/* Video Preview Card */}
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
                      <span className="text-xs text-muted-foreground font-medium">{getVideoPlatformLabel(url)}</span>
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

            {/* Non-YouTube URL Preview */}
            {url && !videoId && !urlError && (
              <div className="relative rounded-lg overflow-hidden border border-border/50 bg-secondary/30">
                <div className="aspect-video relative">
                  <div className="w-full h-full bg-gradient-to-br from-secondary via-secondary/80 to-secondary/60 flex flex-col items-center justify-center">
                    <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center mb-3 border border-primary/30">
                      <Play className="w-8 h-8 text-primary ml-1" />
                    </div>
                    <span className="text-xs text-muted-foreground font-medium">{getVideoPlatformLabel(url)}</span>
                  </div>
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

            {/* Desktop App Reference - Prominent placement */}
            <div className="rounded-lg border border-primary/30 bg-primary/5 p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Monitor className="w-5 h-5 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground">
                    {t('preferLocalExtraction')}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {t('useDesktopApp')}
                  </p>
                </div>
                <a
                  href="https://github.com/jonatl5/guitar-tab-desktop"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors"
                >
                  GitHub
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>

            {/* Advanced Options Collapsible */}
            <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
              <CollapsibleTrigger asChild>
                <Button 
                  variant="ghost" 
                  className="w-full justify-between text-muted-foreground hover:text-foreground h-10"
                >
                  <span className="flex items-center gap-2 text-sm">
                    <Cookie className="w-4 h-4" />
                    {t('advancedOptions')}
                  </span>
                  <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${showAdvanced ? 'rotate-180' : ''}`} />
                </Button>
              </CollapsibleTrigger>
              
              <CollapsibleContent className="space-y-4 pt-2">
                {/* Restricted Site Warning */}
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-amber-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-medium text-foreground mb-1">
                        {t('restrictedSiteWarning')}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {t('restrictedSiteDesc')}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Use Cookies Toggle */}
                <div className="flex items-center justify-between rounded-lg border border-border/50 bg-secondary/30 p-4">
                  <div className="flex items-center gap-3">
                    <Shield className="w-5 h-5 text-muted-foreground" />
                    <Label htmlFor="use-cookies" className="text-sm font-medium cursor-pointer">
                      {t('useCustomCookies')}
                    </Label>
                  </div>
                  <Switch
                    id="use-cookies"
                    checked={useCookies}
                    onCheckedChange={setUseCookies}
                  />
                </div>

                {/* Cookies Textarea */}
                {useCookies && (
                  <div className="space-y-3">
                    <div className="relative">
                      <Textarea
                        placeholder={t('cookiesPlaceholder')}
                        value={cookiesText}
                        onChange={(e) => setCookiesText(e.target.value)}
                        className="min-h-[120px] bg-secondary/50 border-border/50 focus:border-primary/50 font-mono text-xs"
                        disabled={isProcessing}
                      />
                      {cookiesText && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="absolute top-2 right-2 h-7 text-xs text-muted-foreground hover:text-foreground"
                          onClick={handleClearCookies}
                          disabled={isProcessing}
                        >
                          {t('clearCookies')}
                        </Button>
                      )}
                    </div>

                    {/* Privacy Note */}
                    <div className="flex items-start gap-2 text-xs text-muted-foreground">
                      <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                      <p>{t('cookiesPrivacyNote')}</p>
                    </div>
                  </div>
                )}

                {/* Cookies Tutorial Accordion */}
                <Accordion type="single" collapsible className="border border-border/50 rounded-lg overflow-hidden">
                  <AccordionItem value="cookies-help" className="border-none">
                    <AccordionTrigger className="px-4 py-3 hover:no-underline hover:bg-secondary/30 text-sm">
                      {t('howToUseCookies')}
                    </AccordionTrigger>
                    <AccordionContent className="px-4 pb-4">
                      <ol className="space-y-2 text-sm text-muted-foreground list-decimal list-inside">
                        <li>{t('cookiesStep1')}</li>
                        <li>{t('cookiesStep2')}</li>
                        <li>{t('cookiesStep3')}</li>
                        <li>{t('cookiesStep4')}</li>
                        <li>{t('cookiesStep5')}</li>
                      </ol>
                      
                      {/* Caution Note */}
                      <div className="mt-4 rounded-lg border border-destructive/30 bg-destructive/5 p-3">
                        <div className="flex items-start gap-2">
                          <AlertTriangle className="w-4 h-4 text-destructive mt-0.5 flex-shrink-0" />
                          <div>
                            <p className="text-xs font-medium text-destructive mb-0.5">
                              {t('cookiesCaution')}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {t('cookiesCautionText')}
                            </p>
                          </div>
                        </div>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>

              </CollapsibleContent>
            </Collapsible>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
