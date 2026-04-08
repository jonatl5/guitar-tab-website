'use client';

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';

export type Locale = 'en' | 'zh';

// Translation dictionary
export const translations = {
  en: {
    // Header
    appTitle: 'Guitar Tab Extractor',
    appSubtitle: 'Extract guitar tab screenshots from videos',
    connecting: 'Connecting...',
    connected: 'Connected',
    backendOffline: 'Backend Offline',

    // Source Input
    videoSource: 'Video Source',
    youtubeUrl: 'Video URL',
    uploadVideo: 'Upload Video',
    paste: 'Paste',
    invalidVideoUrl: 'Please enter a valid video URL',
    extractTabs: 'Extract Tabs',
    extracting: 'Extracting...',
    uploading: 'Uploading...',
    processing: 'Processing...',
    uploadProgress: 'Upload Progress',
    dropVideoHere: 'Drop video file here',
    orClickToSelect: 'or click to select file (MP4, WebM, MOV, AVI)',
    youtubeVideo: 'YouTube Video',
    videoThumbnailAlt: 'Video thumbnail',

    // Processing State
    processingVideo: 'Processing Video',
    downloadingFromYoutube: 'Downloading and analyzing video from URL...',
    analyzingUploadedVideo: 'Analyzing uploaded video...',
    processingTimeNote: 'This may take a few minutes depending on video length',
    stepDownload: 'Download Video',
    stepAnalyze: 'Analyze Frames',
    stepExtract: 'Extract Tabs',
    stepUploading: 'Uploading',
    stepProcessing: 'Processing',

    // Screenshot Gallery
    extractionResults: 'Extraction Results',
    countPhotos: 'photos',
    selectAll: 'Select All',
    clear: 'Clear',
    tabAt: 'Tab at',

    // Preview Dialog
    tabPreview: 'Tab Preview',
    timestamp: 'Timestamp',
    close: 'Close',
    deselectThis: 'Deselect',
    selectThis: 'Select This Tab',

    // Workflow Sidebar
    workflow: 'Workflow',
    stepSelectVideo: 'Select Video',
    stepExtractTabs: 'Extract Tabs',
    stepSelectScreenshots: 'Select Screenshots',
    stepDownloadPdf: 'Download PDF',
    current: 'Current',
    selected: 'Selected',
    createPdf: 'Create PDF',
    creatingPdf: 'Creating...',
    downloadPdf: 'Download PDF',
    processNewVideo: 'Process New Video',
    helpYoutubeTitle: 'Video URL',
    helpYoutubeDesc: 'Paste a public video link from YouTube, Bilibili, or another yt-dlp supported site and we will try to download and analyze it automatically.',
    helpUploadTitle: 'Local Upload',
    helpUploadDesc: 'Upload MP4, WebM, or other video formats.',
    helpSessionNote: 'Session data is stored in server memory. If you refresh the page, you will need to reprocess the video.',

    // Empty State
    readyToExtract: 'Ready to Extract Tabs',
    emptyStateDesc: 'Enter a video link or upload a local video file to start extracting guitar tab screenshots',

    // Error State
    processingFailed: 'Processing Failed',
    retry: 'Retry',

    // Zero Results State
    noTabsDetected: 'No Guitar Tabs Detected',
    zeroResultsDesc: 'We analyzed the video but could not find any guitar tab screenshots. This might happen if:',
    zeroResultsReason1: 'The video does not contain guitar tabs',
    zeroResultsReason2: 'The tab images are not clear enough',
    zeroResultsReason3: 'The tabs appear in an unusual format',
    tryAnotherVideo: 'Try Another Video',
    uploadLocalVideo: 'Upload Local Video',
    startOver: 'Start Over',

    // Session Expired
    sessionExpired: 'Session Expired',
    sessionExpiredDesc: 'Your session has expired or was not found. Sessions are temporary and stored in server memory.',
    reprocessVideo: 'Reprocess Video',

    // Success State
    pdfReady: 'PDF Ready!',
    pdfReadyDesc: 'Your guitar tab collection has been created successfully.',
    pdfContains: 'Contains {count} selected screenshots',

    // Toast Messages
    toastSuccess: 'Successfully extracted {count} guitar tab screenshots',
    toastPdfSuccess: 'PDF created successfully',
    toastPdfDownloadStarted: 'PDF download started',
    toastSessionExpired: 'Session expired, please reprocess the video',
    toastError: 'An error occurred',
    errorUnknown: 'An unknown error occurred',
    errorBackendUnavailable: 'Backend is not available',
    errorProcessFailed: 'Failed to process video',
    errorParseResponse: 'Failed to parse server response',
    errorNetwork: 'Network error occurred',
    errorCreatePdfFailed: 'Failed to create PDF',
  },
  zh: {
    // Header
    appTitle: 'Guitar Tab Extractor',
    appSubtitle: '从视频中提取吉他谱截图',
    connecting: '连接中...',
    connected: '已连接',
    backendOffline: '后端离线',

    // Source Input
    videoSource: '视频来源',
    youtubeUrl: '视频链接',
    uploadVideo: '上传视频',
    paste: '粘贴',
    invalidVideoUrl: '请输入有效的视频链接',
    extractTabs: '提取吉他谱',
    extracting: '提取中...',
    uploading: '上传中...',
    processing: '处理中...',
    uploadProgress: '上传进度',
    dropVideoHere: '拖放视频文件到这里',
    orClickToSelect: '或点击选择文件 (MP4, WebM, MOV, AVI)',
    youtubeVideo: 'YouTube 视频',
    videoThumbnailAlt: '视频缩略图',

    // Processing State
    processingVideo: '正在处理视频',
    downloadingFromYoutube: '正在下载并分析视频链接...',
    analyzingUploadedVideo: '正在分析上传的视频...',
    processingTimeNote: '这可能需要几分钟时间，具体取决于视频长度',
    stepDownload: '下载视频',
    stepAnalyze: '分析帧',
    stepExtract: '提取吉他谱',
    stepUploading: '上传中',
    stepProcessing: '处理中',

    // Screenshot Gallery
    extractionResults: '提取结果',
    countPhotos: '张',
    selectAll: '全选',
    clear: '清除',
    tabAt: '吉他谱位于',

    // Preview Dialog
    tabPreview: '吉他谱预览',
    timestamp: '时间戳',
    close: '关闭',
    deselectThis: '取消选择',
    selectThis: '选择此谱',

    // Workflow Sidebar
    workflow: '工作流程',
    stepSelectVideo: '选择视频',
    stepExtractTabs: '提取吉他谱',
    stepSelectScreenshots: '选择截图',
    stepDownloadPdf: '下载 PDF',
    current: '当前',
    selected: '已选择',
    createPdf: '创建 PDF',
    creatingPdf: '生成中...',
    downloadPdf: '下载 PDF',
    processNewVideo: '处理新视频',
    helpYoutubeTitle: '视频链接',
    helpYoutubeDesc: '粘贴来自 YouTube、Bilibili 或其他 yt-dlp 支持网站的公开视频链接，我们会尽力自动下载并分析。',
    helpUploadTitle: '本地上传',
    helpUploadDesc: '上传 MP4、WebM 或其他视频格式。',
    helpSessionNote: '会话数据保存在服务器内存中。如果刷新页面，需要重新处理视频。',

    // Empty State
    readyToExtract: '准备提取吉他谱',
    emptyStateDesc: '输入视频链接或上传本地视频文件，开始提取吉他谱截图',

    // Error State
    processingFailed: '处理失败',
    retry: '重试',

    // Zero Results State
    noTabsDetected: '未检测到吉他谱',
    zeroResultsDesc: '我们分析了视频，但未能找到任何吉他谱截图。可能的原因：',
    zeroResultsReason1: '视频中不包含吉他谱',
    zeroResultsReason2: '谱面图像不够清晰',
    zeroResultsReason3: '吉他谱的格式不常见',
    tryAnotherVideo: '尝试其他视频',
    uploadLocalVideo: '上传本地视频',
    startOver: '重新开始',

    // Session Expired
    sessionExpired: '会话已过期',
    sessionExpiredDesc: '您的会话已过期或未找到。会话是临时的，保存在服务器内存中。',
    reprocessVideo: '重新处理视频',

    // Success State
    pdfReady: 'PDF 已就绪！',
    pdfReadyDesc: '您的吉他谱合集已成功创建。',
    pdfContains: '包含 {count} 张选定的截图',

    // Toast Messages
    toastSuccess: '成功提取 {count} 张吉他谱截图',
    toastPdfSuccess: 'PDF 创建成功',
    toastPdfDownloadStarted: 'PDF 下载已开始',
    toastSessionExpired: '会话已过期，请重新处理视频',
    toastError: '发生错误',
    errorUnknown: '发生未知错误',
    errorBackendUnavailable: '后端当前不可用',
    errorProcessFailed: '视频处理失败',
    errorParseResponse: '无法解析服务器响应',
    errorNetwork: '网络错误，请稍后重试',
    errorCreatePdfFailed: '创建 PDF 失败',
  },
} as const;

export type TranslationKey = keyof typeof translations.en;

// Context
type I18nContextType = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
};

const I18nContext = createContext<I18nContextType | null>(null);

// Provider - hydration-safe: renders immediately with default locale, then updates
export function I18nProvider({ children }: { children: ReactNode }) {
  // Start with 'en' as default to match server render
  const [locale, setLocaleState] = useState<Locale>('en');
  const [mounted, setMounted] = useState(false);

  // After mount, check localStorage and browser language
  useEffect(() => {
    setMounted(true);
    const saved = localStorage.getItem('locale') as Locale | null;
    if (saved && (saved === 'en' || saved === 'zh')) {
      setLocaleState(saved);
    } else {
      // Detect browser language
      const browserLang = navigator.language || (navigator as { userLanguage?: string }).userLanguage || 'en';
      if (browserLang.startsWith('zh')) {
        setLocaleState('zh');
        localStorage.setItem('locale', 'zh');
      }
    }
  }, []);

  // Update document lang attribute when locale changes (only after mount)
  useEffect(() => {
    if (mounted) {
      document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en';
    }
  }, [locale, mounted]);

  const setLocale = useCallback((newLocale: Locale) => {
    setLocaleState(newLocale);
    localStorage.setItem('locale', newLocale);
  }, []);

  const t = useCallback((key: TranslationKey, params?: Record<string, string | number>): string => {
    let text = translations[locale][key] || translations.en[key] || key;
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        text = text.replace(`{${k}}`, String(v));
      });
    }
    return text;
  }, [locale]);

  // Always render children - use default 'en' before hydration, then switch
  // This prevents blank screen while still allowing locale to update
  return (
    <I18nContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </I18nContext.Provider>
  );
}

// Hook
export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
}
