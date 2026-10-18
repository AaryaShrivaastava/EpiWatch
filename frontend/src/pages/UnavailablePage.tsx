import { Info } from 'lucide-react';

export default function UnavailablePage({ title }: { title: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full pt-20">
      <div className="bg-gray-100 p-4 rounded-full mb-4">
        <Info size={40} className="text-gray-400" />
      </div>
      <h2 className="text-2xl font-bold text-gray-700 mb-2">{title} is Unavailable</h2>
      <p className="text-gray-500 max-w-md text-center">
        The underlying analytical module or API endpoint for this feature is not currently available or fully implemented in the backend. 
      </p>
    </div>
  );
}
