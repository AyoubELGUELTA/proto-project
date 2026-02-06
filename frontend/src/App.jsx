import React, { useState, useEffect, useRef } from 'react';
import { ragApi } from './api/ragClient';
import { Send, Upload, Trash2, Loader2, User, Bot } from 'lucide-react';

const SourceAccordion = ({ sources }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedSource, setExpandedSource] = useState(null);
  
  return (
    <div className="mt-4 pt-4 border-t border-gray-100">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="text-[10px] font-bold uppercase tracking-widest text-gray-400 hover:text-blue-600 flex items-center gap-1 transition-colors"
      >
        {isOpen ? '⬇ Cacher les détails techniques' : `➡ Voir les ${sources.length} sources analysées`}
      </button>
      
      {isOpen && (
        <div className="mt-2 space-y-2 animate-in fade-in slide-in-from-top-1 duration-200">
          {sources.map((source, idx) => {
            const isExpanded = expandedSource === idx;
            const isIdentity = source.is_identity;

            return (
              <div 
                key={idx} 
                onClick={() => setExpandedSource(isExpanded ? null : idx)}
                className={`text-[11px] p-2 rounded border transition-all cursor-pointer ${
                  isIdentity ? 'bg-blue-50 border-blue-200' : 'bg-gray-50 border-gray-100'
                } ${isExpanded ? 'shadow-md' : 'hover:bg-gray-100'}`}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className={`font-bold text-[9px] uppercase ${isIdentity ? 'text-blue-600' : 'text-gray-400'}`}>
                    {isIdentity ? '📑 Fiche Identité' : `📦 Chunk #${source.chunk_index || idx}`}
                  </span>
                  <span className="text-[8px] text-gray-400">{isExpanded ? 'Réduire ▲' : 'Détails ▼'}</span>
                </div>
                
                {/* Titre du chunk ou nom du doc */}
                <p className="font-semibold text-gray-700 truncate">
                  {source.heading_full || "Sans titre"}
                </p>

                {isExpanded && (
                  <div className="mt-2 space-y-3 pt-2 border-t border-gray-200 animate-in zoom-in-95 duration-150">
                    
                    {/* LE TEXTE ENVOYÉ AU RERANKER (Le plus important) */}
                    <div>
                      <span className="text-[8px] font-bold text-red-400 uppercase">Input Reranker / LLM :</span>
                      <div className="mt-1 p-2 bg-gray-900 text-green-400 font-mono text-[10px] rounded leading-tight">
                        {source.text_for_reranker || source.text}
                      </div>
                    </div>

                    {/* IMAGES & VISUAL SUMMARY */}
                    {(source.images_urls?.length > 0 || source.visual_summary) && (
                      <div className="space-y-2">
                        <span className="text-[8px] font-bold text-purple-500 uppercase">Analyse Visuelle :</span>
                        {source.images_urls?.map((url, i) => (
                          <img key={i} src={url} alt="Source" className="max-w-full rounded border border-purple-200" />
                        ))}
                        {source.visual_summary && (
                          <p className="p-2 bg-purple-50 text-purple-700 rounded italic text-[10px]">
                            "{source.visual_summary}"
                          </p>
                        )}
                      </div>
                    )}

                    {/* TABLES BRUTES */}
                    {source.tables && source.tables.length > 0 && (
                      <div>
                        <span className="text-[8px] font-bold text-orange-500 uppercase">Données Tabulaires :</span>
                        <div className="mt-1 overflow-x-auto p-1 bg-orange-50 text-orange-800 rounded text-[9px]">
                          <pre>{JSON.stringify(source.tables, null, 2)}</pre>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

function App() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]); // { role: 'user'|'ai', text: '', sources: [] }
  const [isUploading, setIsUploading] = useState(false);
  const [isAnswerLoading, setIsAnswerLoading] = useState(false); 
  const [ingestedFiles, setIngestedFiles] = useState([]);
  const lastUserMsgRef = useRef(null);
  const lastAiMsgRef = useRef(null);


  useEffect(() => { // To Scroll to the last message
  if (messages.length === 0) return;

  const lastMessage = messages[messages.length - 1];

  if (lastMessage.role === 'user') {
    // 1. On cadre la question de l'utilisateur en haut
    lastUserMsgRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  } 
  else if (lastMessage.role === 'ai') {
    // 2. Dès que l'IA répond, on glisse doucement vers sa réponse
    // On attend un tout petit délai (100ms) pour laisser le DOM s'ajuster
    setTimeout(() => {
      lastAiMsgRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  }
}, [messages]);


  useEffect(() => { //To Reset history chat every time page is refreshed
    const resetChat = async () => {
      try {
        // On appelle la route de reset de ton gateway
        await ragApi.clearHistory(); 
        console.log("🗑️ Historique nettoyé pour une nouvelle session de test");
      } catch (err) {
        console.error("Impossible de nettoyer l'historique:", err);
      }
    };

    resetChat();
  }, []); // Le tableau vide [] signifie "exécuter une seule fois au chargement"

  const fetchFiles = async () => {
  try {
    const res = await ragApi.getIngestedDocuments();
    console.log("Docs reçus 'res':", res);
    setIngestedFiles(res.data.documents);
  } catch (err) {
    console.error("Erreur lors de la récupération des fichiers", err);
  }
};

  useEffect(() => { //Fetch Files every time we get to the page / or refresh a page.
  fetchFiles();
}, []);

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const userMsg = { role: 'user', text: question };
    setMessages(prev => [...prev, userMsg]);
    setQuestion('');
    setIsAnswerLoading(true);
    try {
      const response = await ragApi.askQuestion(userMsg.text);
      const aiMsg = { 
        role: 'ai', 
        text: response.data.answer, 
        sources: response.data.sources 
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      console.error("Erreur Query:", err);
    }
    finally {
        setIsAnswerLoading(false); // 2. On arrête le loader (Succès ou Échec)
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Optionnel : Petite restriction de sécurité front
    if (file.type !== "application/pdf") {
        alert("Seuls les fichiers PDF sont acceptés.");
        return;
    }

    setIsUploading(true);
    try {
        // On utilise la méthode déjà prête dans ton ragClient
        await ragApi.ingestPdf(file); 
        await fetchFiles()
        alert("C'est bon ! Votre document a été lu et appris !");
    } catch (err) {
        console.error("Erreur Ingestion:", err);
        alert("Erreur lors de l'ingestion.");
    } finally {
        setIsUploading(false);
        // Reset l'input pour pouvoir uploader le même fichier si besoin
        event.target.value = null; 
    }
};

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar - Upload */}
      <div className="w-1/4 bg-white border-r p-6 flex flex-col space-y-6">
  <h1 className="text-xl font-bold text-blue-600">Dawa RAG</h1>
  <div className="border-2 border-dashed border-gray-200 rounded-lg p-4 text-center">
    <p className="text-sm text-gray-500 mb-4">Ingérer un nouveau document</p>
    
    {/* On utilise un label pour styliser l'input file invisible */}
    <label className={`flex items-center justify-center w-full py-2 rounded-md transition cursor-pointer ${
      isUploading ? 'bg-gray-100 text-gray-400' : 'bg-blue-50 text-blue-600 hover:bg-blue-100'
    }`}>
      {isUploading ? (
        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
      ) : (
        <Upload className="w-4 h-4 mr-2" />
      )}
      
      <span>{isUploading ? 'Traitement...' : 'Sélectionner'}</span>

      {/* L'input réel est caché mais activé par le clic sur le label */}
      <input 
        type="file" 
        className="hidden" 
        onChange={handleFileUpload} 
        accept=".pdf"
        disabled={isUploading} 
      />
    </label>
    
    {isUploading && (
      <p className="text-[10px] text-blue-500 mt-2 animate-pulse">
        Vectorisation en cours...
      </p>
    )}
  </div>

  {/*Documents ingérés par la rag deja*/}

<div className="mt-8">
  <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-4">
    {/* On utilise || 0 pour éviter le crash si length est inaccessible */}
    Documents indexés ({(ingestedFiles || []).length})
  </h2>
  
  <div className="space-y-2 overflow-y-auto max-h-64 pr-2">
    {/* On s'assure que ingestedFiles est traité comme un tableau */}
    {(ingestedFiles || []).map((filename, idx) => (
      <div key={idx} className="flex items-center p-2 bg-gray-50 rounded-md border border-gray-100 group">
        <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded flex items-center justify-center mr-3 text-[10px] font-bold">
          PDF
        </div>
        <span className="text-xs text-gray-600 truncate flex-1" title={filename}>
          {filename}
        </span>
      </div>
    ))}
    
    {/* Message si vide */}
    {(!ingestedFiles || ingestedFiles.length === 0) && (
      <p className="text-xs text-gray-400 italic">Aucun document pour le moment.</p>
    )}
  </div>
</div>

</div>





      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 p-6 overflow-y-auto space-y-4">
          {messages.map((m, i) => {
    // On vérifie si c'est le TOUT DERNIER message de la liste et s'il vient du 'user'
    const isLastMessage = i === messages.length - 1;  
    const isAi = m.role === 'ai';  
    return (
    <div 
      key={i} 
      ref={isLastMessage ? (isAi ? lastAiMsgRef : lastUserMsgRef) : null}
      className={`flex items-start gap-3 ${isAi ? 'flex-row' : 'flex-row-reverse'}`}
    >
      {/* Icône Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center border ${
        isAi ? 'bg-white border-gray-200 text-blue-600' : 'bg-blue-600 border-blue-500 text-white'
      }`}>
        {isAi ? <Bot size={18} /> : <User size={18} />}
      </div>

      {/* Bulle de message */}
      <div className={`max-w-2xl p-4 rounded-2xl shadow-sm ${
        isAi 
          ? 'bg-white border border-gray-100 text-gray-800 rounded-tl-none' 
          : 'bg-blue-600 text-white rounded-tr-none'
      }`}>
        <p className="leading-relaxed">{m.text}</p>
        
        {isAi && m.sources && m.sources.length > 0 && (
          <SourceAccordion sources={m.sources} />
        )}
      </div>
    </div>
  );
})}
          {isAnswerLoading && (
    <div className="flex items-center space-x-2 text-gray-400 text-sm animate-pulse p-4">
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
      </div>
      <span>Analyse de vos documents...</span>
    </div>
  )}
        </div>

        {/* Input area */}
        <form onSubmit={handleQuery} className="p-4 bg-white border-t flex gap-4">
          <input 
            className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Comment puis-je vous aider ?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button type="submit" className="bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700">
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;