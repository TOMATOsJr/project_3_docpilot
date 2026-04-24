import { useEffect, useMemo, useRef, useState } from 'react';

type DocumentRecord = {
  document: {
    id: string;
    filename: string;
    document_type: string;
    uploaded_at: string;
  };
  chunk_count: number;
};

type QueryResponse = {
  answer: string;
  citations: Array<{
    document_id: string;
    source_label: string | null;
    quote: string | null;
  }>;
  model_used: string;
  model_selection_reason: string | null;
  fallback_used: boolean;
};

type DiffLine = {
  kind: 'insert' | 'delete' | 'equal';
  content: string;
};

type EditResponse = {
  proposal: {
    command_id: string;
    document_id: string;
    diff: DiffLine[];
  };
  model_used: string | null;
  model_selection_reason: string | null;
  fallback_used: boolean;
  status: 'pending' | 'applied' | 'rejected';
};

type ChatMode = 'qa' | 'edit';
type UploadState = 'idle' | 'uploading' | 'error';

type ChatTurn = {
  id: string;
  role: 'user' | 'assistant';
  mode: ChatMode;
  message: string;
};

type FileContentResponse = {
  document_id: string;
  filename: string;
  document_type: string;
  content: string;
};

export default function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [activeDocumentId, setActiveDocumentId] = useState<string>('');
  const [activeFilename, setActiveFilename] = useState<string>('');
  const [activeDocumentType, setActiveDocumentType] = useState<string>('');
  const [activeContent, setActiveContent] = useState<string>('');
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [notice, setNotice] = useState<string>('');

  const [mode, setMode] = useState<ChatMode>('qa');
  const [chatInput, setChatInput] = useState('');
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('auto');
  const [useActiveOnlyForQa, setUseActiveOnlyForQa] = useState(true);

  const [pendingEdit, setPendingEdit] = useState<EditResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const sortedDocuments = useMemo(
    () =>
      [...documents].sort((left, right) => {
        const leftDate = new Date(left.document.uploaded_at).getTime();
        const rightDate = new Date(right.document.uploaded_at).getTime();
        return rightDate - leftDate;
      }),
    [documents],
  );

  const centerDiff = useMemo(() => {
    if (!pendingEdit) {
      return [] as DiffLine[];
    }
    if (pendingEdit.proposal.document_id !== activeDocumentId) {
      return [] as DiffLine[];
    }
    return pendingEdit.proposal.diff;
  }, [pendingEdit, activeDocumentId]);

  useEffect(() => {
    void loadDocuments();
    void loadAvailableModels();
  }, []);

  useEffect(() => {
    if (!sortedDocuments.length) {
      setActiveDocumentId('');
      setActiveFilename('');
      setActiveDocumentType('');
      setActiveContent('');
      return;
    }

    if (!sortedDocuments.some((doc) => doc.document.id === activeDocumentId)) {
      const first = sortedDocuments[0];
      setActiveDocumentId(first.document.id);
    }
  }, [sortedDocuments, activeDocumentId]);

  useEffect(() => {
    if (!activeDocumentId) {
      return;
    }
    void loadDocumentContent(activeDocumentId);
  }, [activeDocumentId]);

  async function loadDocuments() {
    const response = await fetch('/api/documents');
    if (!response.ok) {
      throw new Error('Failed to load documents');
    }
    const payload = (await response.json()) as DocumentRecord[];
    setDocuments(payload);
  }

  async function loadAvailableModels() {
    try {
      const response = await fetch('/api/qa/models');
      if (!response.ok) {
        throw new Error('Failed to load models');
      }
      const payload = (await response.json()) as { available_models: string[] };
      setAvailableModels(payload.available_models);
    } catch {
      setAvailableModels([]);
    }
  }

  async function loadDocumentContent(documentId: string) {
    const response = await fetch(`/api/documents/${documentId}/content`);
    if (!response.ok) {
      setActiveContent('');
      return;
    }

    const payload = (await response.json()) as FileContentResponse;
    setActiveFilename(payload.filename);
    setActiveDocumentType(payload.document_type);
    setActiveContent(payload.content || '');
  }

  async function handleFileUpload(files: FileList | null) {
    if (!files || files.length === 0) {
      return;
    }

    setUploadState('uploading');
    setNotice(`Uploading ${files.length} file${files.length > 1 ? 's' : ''}...`);

    try {
      for (const file of Array.from(files)) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/documents/upload', {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
      }
      setUploadState('idle');
      setNotice('Upload complete.');
      await loadDocuments();
    } catch (error) {
      setUploadState('error');
      setNotice(error instanceof Error ? error.message : 'Upload failed.');
    }
  }

  async function submitChat(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const input = chatInput.trim();
    if (!input) {
      return;
    }

    const userTurn: ChatTurn = {
      id: crypto.randomUUID(),
      role: 'user',
      mode,
      message: input,
    };
    setChatTurns((current) => [userTurn, ...current]);
    setChatInput('');
    setIsSubmitting(true);

    try {
      if (mode === 'qa') {
        await submitQa(input);
      } else {
        await submitEdit(input);
      }
    } catch (error) {
      const errorTurn: ChatTurn = {
        id: crypto.randomUUID(),
        role: 'assistant',
        mode,
        message: error instanceof Error ? error.message : 'Request failed.',
      };
      setChatTurns((current) => [errorTurn, ...current]);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function submitQa(input: string) {
    const documentIds = useActiveOnlyForQa && activeDocumentId ? [activeDocumentId] : [];

    const response = await fetch('/api/qa', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: input,
        document_ids: documentIds,
        requested_model: selectedModel === 'auto' ? null : selectedModel,
        max_chunks: 5,
        conversation_history: [],
      }),
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const payload = (await response.json()) as QueryResponse;
    const citationSummary = payload.citations
      .slice(0, 2)
      .map((c) => `${c.source_label || 'source'}: ${c.quote || 'no quote'}`)
      .join(' | ');

    const assistantTurn: ChatTurn = {
      id: crypto.randomUUID(),
      role: 'assistant',
      mode: 'qa',
      message: `${payload.answer}\n\nModel: ${payload.model_used}${payload.fallback_used ? ' (fallback)' : ''}${payload.model_selection_reason ? `\nWhy: ${payload.model_selection_reason}` : ''}${citationSummary ? `\nCitations: ${citationSummary}` : ''}`,
    };
    setChatTurns((current) => [assistantTurn, ...current]);
  }

  async function submitEdit(instruction: string) {
    if (!activeDocumentId) {
      throw new Error('Pick a file first.');
    }

    const response = await fetch('/api/edits/propose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        document_id: activeDocumentId,
        instruction,
        requested_model: selectedModel === 'auto' ? null : selectedModel,
      }),
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const payload = (await response.json()) as EditResponse;
    setPendingEdit(payload);

    const assistantTurn: ChatTurn = {
      id: crypto.randomUUID(),
      role: 'assistant',
      mode: 'edit',
      message: `Edit proposal ready. Review diff in center and choose Accept or Reject.\nCommand: ${payload.proposal.command_id}${payload.model_used ? `\nModel: ${payload.model_used}${payload.fallback_used ? ' (fallback)' : ''}` : ''}${payload.model_selection_reason ? `\nWhy: ${payload.model_selection_reason}` : ''}`,
    };
    setChatTurns((current) => [assistantTurn, ...current]);
  }

  async function resolveEdit(action: 'apply' | 'reject') {
    if (!pendingEdit) {
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await fetch(`/api/edits/${pendingEdit.proposal.command_id}/${action}`, { method: 'POST' });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload = (await response.json()) as EditResponse;
      setPendingEdit(payload);

      if (action === 'apply') {
        await loadDocumentContent(payload.proposal.document_id);
      }

      const assistantTurn: ChatTurn = {
        id: crypto.randomUUID(),
        role: 'assistant',
        mode: 'edit',
        message: action === 'apply' ? 'Edit applied. You can now download the edited file.' : 'Edit rejected.',
      };
      setChatTurns((current) => [assistantTurn, ...current]);
    } catch (error) {
      const assistantTurn: ChatTurn = {
        id: crypto.randomUUID(),
        role: 'assistant',
        mode: 'edit',
        message: error instanceof Error ? error.message : 'Failed to resolve proposal.',
      };
      setChatTurns((current) => [assistantTurn, ...current]);
    } finally {
      setIsSubmitting(false);
    }
  }

  function downloadActiveDocument() {
    if (!activeDocumentId) {
      return;
    }
    window.open(`/api/documents/${activeDocumentId}/download`, '_blank');
  }

  function clearPendingDiff() {
    setPendingEdit(null);
  }

  return (
    <div className="copilotShell">
      <aside className="pane leftPane">
        <div className="paneTitle">Workspace</div>
        <button type="button" className="paneButton" onClick={() => fileInputRef.current?.click()} disabled={uploadState === 'uploading'}>
          {uploadState === 'uploading' ? 'Uploading...' : 'Upload Documents'}
        </button>
        <input
          ref={fileInputRef}
          className="hiddenInput"
          type="file"
          multiple
          accept=".pdf,.docx,.pptx,.md,.txt"
          onChange={(event) => {
            void handleFileUpload(event.target.files);
            event.target.value = '';
          }}
        />
        {notice ? <p className="noticeLine">{notice}</p> : null}

        <div className="fileList">
          {sortedDocuments.map((record) => {
            const isActive = record.document.id === activeDocumentId;
            return (
              <button
                key={record.document.id}
                type="button"
                className={`fileItem ${isActive ? 'active' : ''}`}
                onClick={() => setActiveDocumentId(record.document.id)}
              >
                <strong>{record.document.filename}</strong>
                <small>{record.document.document_type} · {record.chunk_count} chunks</small>
              </button>
            );
          })}
        </div>
      </aside>

      <section className="pane centerPane">
        <div className="centerHeader">
          <div>
            <h2>{activeFilename || 'No file selected'}</h2>
            <small>{activeDocumentType || 'Select a file to view full content'}</small>
          </div>
          <div className="centerActions">
            <button type="button" className="paneButton" onClick={() => void loadDocumentContent(activeDocumentId)} disabled={!activeDocumentId}>
              Refresh View
            </button>
            <button type="button" className="paneButton" onClick={downloadActiveDocument} disabled={!activeDocumentId}>
              Download Edited File
            </button>
          </div>
        </div>

        {centerDiff.length > 0 && pendingEdit?.status === 'pending' ? (
          <div className="diffBanner">
            <span>Pending diff preview</span>
            <button type="button" onClick={clearPendingDiff}>Hide Diff</button>
          </div>
        ) : null}

        <div className="fileViewport">
          {centerDiff.length > 0 && pendingEdit?.status === 'pending' ? (
            <pre className="fileContent">
              {centerDiff.map((line, index) => {
                const klass = line.kind === 'insert' ? 'lineAdd' : line.kind === 'delete' ? 'lineDelete' : 'lineEqual';
                const prefix = line.kind === 'insert' ? '+' : line.kind === 'delete' ? '-' : ' ';
                return (
                  <div key={`${line.kind}-${index}`} className={klass}>
                    {prefix} {line.content}
                  </div>
                );
              })}
            </pre>
          ) : (
            <pre className="fileContent">{activeContent || 'No content to display.'}</pre>
          )}
        </div>
      </section>

      <aside className="pane rightPane">
        <div className="paneTitle">Copilot Chat</div>

        <div className="modeSwitch">
          <button type="button" className={mode === 'qa' ? 'active' : ''} onClick={() => setMode('qa')}>Q&A</button>
          <button type="button" className={mode === 'edit' ? 'active' : ''} onClick={() => setMode('edit')}>Edit</button>
        </div>

        <label className="fieldLabel" htmlFor="modelChoice">Model</label>
        <select id="modelChoice" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
          <option value="auto">Auto (dynamic choice)</option>
          {availableModels.map((model) => (
            <option key={model} value={model}>{model}</option>
          ))}
        </select>
        <small className="noticeLine">
          {selectedModel === 'auto'
            ? 'Auto lets DocPilot pick the best model per task and context size.'
            : 'Manual mode pins this model for the next request.'}
        </small>

        {mode === 'qa' ? (
          <label className="toggleLine">
            <input
              type="checkbox"
              checked={useActiveOnlyForQa}
              onChange={(event) => setUseActiveOnlyForQa(event.target.checked)}
            />
            Scope Q&A to active center file
          </label>
        ) : null}

        {mode === 'edit' && pendingEdit?.status === 'pending' ? (
          <div className="resolveActions">
            <button type="button" onClick={() => void resolveEdit('apply')} disabled={isSubmitting}>Accept</button>
            <button type="button" className="danger" onClick={() => void resolveEdit('reject')} disabled={isSubmitting}>Reject</button>
          </div>
        ) : null}

        <form className="chatForm" onSubmit={submitChat}>
          <textarea
            value={chatInput}
            onChange={(event) => setChatInput(event.target.value)}
            placeholder={mode === 'qa' ? 'Ask about your docs...' : 'Describe the edit you want applied to the active file...'}
            rows={4}
          />
          <button type="submit" disabled={isSubmitting || !chatInput.trim()}>
            {isSubmitting ? 'Working...' : mode === 'qa' ? 'Ask' : 'Propose Edit'}
          </button>
        </form>

        <div className="chatHistory">
          {chatTurns.map((turn) => (
            <article key={turn.id} className={`chatTurn ${turn.role}`}>
              <strong>{turn.role === 'user' ? `You · ${turn.mode.toUpperCase()}` : `DocPilot · ${turn.mode.toUpperCase()}`}</strong>
              <p>{turn.message}</p>
            </article>
          ))}
        </div>
      </aside>
    </div>
  );
}
