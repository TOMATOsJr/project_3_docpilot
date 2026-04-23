import { useEffect, useMemo, useRef, useState } from 'react';

type HealthResponse = {
  status: string;
  service: string;
  version: string;
  environment: string;
  timestamp: string;
};

type DocumentRecord = {
  document: {
    id: string;
    filename: string;
    document_type: string;
    page_count: number;
    uploaded_at: string;
  };
  chunk_count: number;
};

type UploadState = 'idle' | 'uploading' | 'ready' | 'error';

type Notice = {
  kind: 'success' | 'error' | 'info';
  message: string;
};

type Citation = {
  document_id: string;
  page_number: number | null;
  paragraph_index: number | null;
  source_label: string | null;
  quote: string | null;
};

type QueryResponse = {
  answer: string;
  citations: Citation[];
  retrieved_chunks: Array<{ id: string; text: string }>;
  model_used: string;
};

type QaTurn = {
  id: string;
  query: string;
  response: QueryResponse;
};

function formatDocumentType(documentType: string): string {
  return documentType === 'unknown' ? 'ready' : documentType;
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [notice, setNotice] = useState<Notice | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [query, setQuery] = useState('');
  const [maxChunks, setMaxChunks] = useState(3);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [qaHistory, setQaHistory] = useState<QaTurn[]>([]);
  const [isAsking, setIsAsking] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);

  const sortedDocuments = useMemo(
    () =>
      [...documents].sort((left, right) => {
        const leftDate = new Date(left.document.uploaded_at).getTime();
        const rightDate = new Date(right.document.uploaded_at).getTime();
        return rightDate - leftDate;
      }),
    [documents],
  );

  const documentNameById = useMemo(() => {
    const map = new Map<string, string>();
    for (const doc of documents) {
      map.set(doc.document.id, doc.document.filename);
    }
    return map;
  }, [documents]);

  async function loadHealth() {
    try {
      const response = await fetch('/api/health');
      if (!response.ok) {
        throw new Error('Health check failed');
      }
      const data = (await response.json()) as HealthResponse;
      setHealth(data);
    } catch {
      setHealth(null);
    }
  }

  async function loadDocuments() {
    setRefreshing(true);
    try {
      const response = await fetch('/api/documents');
      if (!response.ok) {
        throw new Error('Failed to load documents');
      }
      const data = (await response.json()) as DocumentRecord[];
      setDocuments(data);
      setSelectedDocumentIds((current) => current.filter((id) => data.some((record) => record.document.id === id)));
    } catch (error) {
      setNotice({ kind: 'error', message: error instanceof Error ? error.message : 'Could not load documents.' });
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    void loadHealth();
    void loadDocuments();
  }, []);

  async function uploadFile(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/documents/upload', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Upload failed for ${file.name}`);
    }
  }

  async function handleFiles(files: FileList | File[]) {
    const validFiles = Array.from(files).filter((file) => file.size > 0);

    if (validFiles.length === 0) {
      setNotice({ kind: 'error', message: 'No files were selected.' });
      return;
    }

    setUploadState('uploading');
    setNotice({ kind: 'info', message: `Uploading ${validFiles.length} file${validFiles.length > 1 ? 's' : ''}...` });

    try {
      for (const file of validFiles) {
        await uploadFile(file);
      }
      setUploadState('ready');
      setNotice({ kind: 'success', message: `Uploaded ${validFiles.length} file${validFiles.length > 1 ? 's' : ''} successfully.` });
      await loadDocuments();
      await loadHealth();
    } catch (error) {
      setUploadState('error');
      setNotice({ kind: 'error', message: error instanceof Error ? error.message : 'Upload failed.' });
    }
  }

  async function deleteDocument(documentId: string, filename: string) {
    const confirmed = window.confirm(`Delete ${filename}?`);
    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(`/api/documents/${documentId}`, {
        method: 'DELETE',
      });

      if (!response.ok && response.status !== 204) {
        const detail = await response.text();
        throw new Error(detail || `Delete failed for ${filename}`);
      }

      setNotice({ kind: 'success', message: `${filename} removed.` });
      await loadDocuments();
      setQaHistory((history) =>
        history.map((turn) => ({
          ...turn,
          response: {
            ...turn.response,
            citations: turn.response.citations.filter((citation) => citation.document_id !== documentId),
          },
        })),
      );
    } catch (error) {
      setNotice({ kind: 'error', message: error instanceof Error ? error.message : 'Delete failed.' });
    }
  }

  async function askQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuery = query.trim();
    if (trimmedQuery.length === 0) {
      setNotice({ kind: 'error', message: 'Please enter a question.' });
      return;
    }

    if (documents.length === 0) {
      setNotice({ kind: 'error', message: 'Upload at least one document before asking a question.' });
      return;
    }

    setIsAsking(true);
    try {
      const response = await fetch('/api/qa', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: trimmedQuery,
          document_ids: selectedDocumentIds,
          max_chunks: maxChunks,
        }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || 'Failed to get answer.');
      }

      const payload = (await response.json()) as QueryResponse;
      const turn: QaTurn = {
        id: crypto.randomUUID(),
        query: trimmedQuery,
        response: payload,
      };
      setQaHistory((history) => [turn, ...history]);
      setQuery('');
      setNotice({ kind: 'success', message: 'Grounded answer generated.' });
    } catch (error) {
      setNotice({ kind: 'error', message: error instanceof Error ? error.message : 'Q&A request failed.' });
    } finally {
      setIsAsking(false);
    }
  }

  function toggleDocumentFilter(documentId: string) {
    setSelectedDocumentIds((current) =>
      current.includes(documentId) ? current.filter((id) => id !== documentId) : [...current, documentId],
    );
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function openFolderPicker() {
    folderInputRef.current?.click();
  }

  function handleDrop(event: React.DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragActive(false);
    void handleFiles(event.dataTransfer.files);
  }

  function handleDragOver(event: React.DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragLeave(event: React.DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragActive(false);
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">D</div>
          <div>
            <h1>DocPilot</h1>
            <p>Document intelligence workspace</p>
          </div>
        </div>

        <section className="panel uploadPanel">
          <h2>Upload</h2>
          <div
            className={`uploadBox ${dragActive ? 'dragActive' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                openFilePicker();
              }
            }}
          >
            <strong>Drop files here</strong>
            <span>PDF, DOCX, PPTX, or MD will be parsed and chunked</span>
            <div className="uploadActions">
              <button type="button" onClick={openFilePicker} disabled={uploadState === 'uploading'}>
                Choose files
              </button>
              <button type="button" onClick={openFolderPicker} disabled={uploadState === 'uploading'}>
                Choose folder
              </button>
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.pptx,.md,.txt"
            className="hiddenInput"
            onChange={(event) => {
              const files = event.target.files;
              if (files && files.length > 0) {
                void handleFiles(files);
              }
              event.target.value = '';
            }}
          />
          <input
            ref={folderInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.pptx,.md,.txt"
            className="hiddenInput"
            {...({ webkitdirectory: '', directory: '' } as React.InputHTMLAttributes<HTMLInputElement> & {
              webkitdirectory?: string;
              directory?: string;
            })}
            onChange={(event) => {
              const files = event.target.files;
              if (files && files.length > 0) {
                void handleFiles(files);
              }
              event.target.value = '';
            }}
          />
          {notice ? <p className={`notice ${notice.kind}`}>{notice.message}</p> : null}
        </section>

        <section className="panel">
          <div className="panelHeader compact">
            <h2>Documents</h2>
            <button type="button" className="textButton" onClick={() => void loadDocuments()}>
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
          <div className="documentList">
            {sortedDocuments.length === 0 ? (
              <div className="emptyState">
                <strong>No documents yet</strong>
                <span>Upload a file or folder to start indexing.</span>
              </div>
            ) : (
              sortedDocuments.map((record) => (
                <article className="documentCard" key={record.document.id}>
                  <div>
                    <strong>{record.document.filename}</strong>
                    <span>
                      {record.chunk_count} chunks • {formatDocumentType(record.document.document_type)}
                    </span>
                  </div>
                  <div className="documentCardActions">
                    <em>{record.chunk_count > 0 ? 'Indexed' : 'Queued'}</em>
                    <button
                      type="button"
                      className="dangerButton"
                      onClick={() => void deleteDocument(record.document.id, record.document.filename)}
                    >
                      Delete
                    </button>
                  </div>
                </article>
              ))
            )}
          </div>
        </section>
      </aside>

      <main className="workspace">
        <header className="hero">
          <div>
            <p className="eyebrow">Copilot-inspired multi-model workflow</p>
            <h2>Ask questions, edit safely, and trace every answer back to source.</h2>
          </div>
          <div className="healthCard">
            <span>Backend</span>
            <strong>{health ? `${health.service} • ${health.status}` : 'Disconnected'}</strong>
            <small>{health ? `${health.version} • ${health.environment}` : 'Waiting for API'}</small>
          </div>
        </header>

        <section className="grid">
          <article className="panel transcriptPanel">
            <div className="panelHeader">
              <h3>Q&A</h3>
              <span>Grounded responses with page citations</span>
            </div>
            <form className="qaForm" onSubmit={askQuestion}>
              <label htmlFor="questionInput">Question</label>
              <textarea
                id="questionInput"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Ask a grounded question about your uploaded documents"
                rows={3}
              />

              <div className="qaControlsRow">
                <label className="fieldInline" htmlFor="maxChunks">
                  Citations to retrieve
                  <input
                    id="maxChunks"
                    type="number"
                    min={1}
                    max={10}
                    value={maxChunks}
                    onChange={(event) => setMaxChunks(Math.max(1, Math.min(10, Number(event.target.value) || 3)))}
                  />
                </label>
                <button type="submit" disabled={isAsking}>
                  {isAsking ? 'Asking...' : 'Ask'}
                </button>
              </div>

              <div className="docFilters">
                <span>Scope</span>
                {sortedDocuments.length === 0 ? (
                  <small>No uploaded documents yet</small>
                ) : (
                  <div className="chipRow">
                    {sortedDocuments.map((record) => {
                      const checked = selectedDocumentIds.includes(record.document.id);
                      return (
                        <label key={record.document.id} className={`chip ${checked ? 'active' : ''}`}>
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleDocumentFilter(record.document.id)}
                          />
                          {record.document.filename}
                        </label>
                      );
                    })}
                  </div>
                )}
                <small>
                  {selectedDocumentIds.length === 0
                    ? 'Using all documents'
                    : `Using ${selectedDocumentIds.length} selected document${selectedDocumentIds.length > 1 ? 's' : ''}`}
                </small>
              </div>
            </form>

            <div className="transcript">
              {qaHistory.length === 0 ? (
                <div className="emptyState">
                  <strong>No questions yet</strong>
                  <span>Ask a question to see grounded answers and citations.</span>
                </div>
              ) : (
                qaHistory.map((turn) => (
                  <div className="qaTurn" key={turn.id}>
                    <div className="bubble user">{turn.query}</div>
                    <div className="bubble assistant">
                      <p>{turn.response.answer}</p>
                      <small>Model: {turn.response.model_used}</small>
                    </div>
                    <div className="citationList">
                      {turn.response.citations.length === 0 ? (
                        <span className="citationEmpty">No citations returned for this answer.</span>
                      ) : (
                        turn.response.citations.map((citation, index) => (
                          <article className="citationCard" key={`${turn.id}-${citation.document_id}-${index}`}>
                            <strong>
                              {citation.source_label || 'Source'} •{' '}
                              {documentNameById.get(citation.document_id) || citation.document_id}
                            </strong>
                            <span>
                              page {citation.page_number ?? 'n/a'} • paragraph {citation.paragraph_index ?? 'n/a'}
                            </span>
                            <p>{citation.quote || 'No quote provided.'}</p>
                          </article>
                        ))
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </article>

          <article className="panel transcriptPanel">
            <div className="panelHeader">
              <h3>Editing</h3>
              <span>Command + Memento with reversible diffs</span>
            </div>
            <div className="diffPreview">
              <div className="diffLine remove">- Replace generic claims with grounded wording</div>
              <div className="diffLine add">+ Replace generic claims with exact cited language</div>
              <div className="diffLine keep">  Keep the original document untouched until accept</div>
            </div>
          </article>
        </section>
      </main>
    </div>
  );
}
