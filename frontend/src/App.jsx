import React, { useState } from 'react';
import { ragApi } from './api/ragClient';
import { Send, Upload, Trash2, Loader2 } from 'lucide-react';

const SourceAccordion = ({ sources }) => {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <div className="mt-4 pt-4 border-t border-gray-100">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="text-[10px] font-bold uppercase tracking-widest text-gray-400 hover:text-blue-600 flex items-center gap-1"
      >
        {isOpen ? '⬇ Cacher les sources' : `➡ Voir les ${sources.length} sources`}
      </button>
      
      {isOpen && (
        <div className="mt-2 space-y-2 animate-in fade-in slide-in-from-top-1">
          {sources.map((source, idx) => (
            <div key={idx} className="text-[11px] bg-gray-50 p-2 rounded border border-gray-100 text-gray-600 italic">
              "{source.text?.substring(0, 200)}..."
            </div>
          ))}
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

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar - Upload */}
      <div className="w-1/4 bg-white border-r p-6 flex flex-col space-y-6">
        <h1 className="text-xl font-bold text-blue-600">Solon RAG</h1>
        <div className="border-2 border-dashed border-gray-200 rounded-lg p-4 text-center">
           <p className="text-sm text-gray-500 mb-4">Ingérer un nouveau document</p>
           {/* On ajoutera le composant FileUploader ici plus tard */}
           <button className="flex items-center justify-center w-full py-2 bg-blue-50 text-blue-600 rounded-md hover:bg-blue-100 transition">
             <Upload className="w-4 h-4 mr-2" /> Sélectionner
           </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 p-6 overflow-y-auto space-y-4">
          {messages.map((m, i) => (
<div className={`max-w-2xl p-4 rounded-lg ${m.role === 'user' ? 'bg-blue-600 text-white ml-auto' : 'bg-white border text-gray-800'}`}>
  <p>{m.text}</p> 
  
  
  {/* Affichage des sources si elles existent */}
  {m.role === 'ai' && m.sources && m.sources.length > 0 && (
    <SourceAccordion sources={m.sources} />
  )}
  
</div>
          ))}
          {isAnswerLoading && (
    <div className="flex items-center space-x-2 text-gray-400 text-sm animate-pulse p-4">
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
      </div>
      <span>Solon analyse vos documents...</span>
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