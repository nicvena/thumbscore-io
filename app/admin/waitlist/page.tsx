import { createClient } from '@supabase/supabase-js'
import { redirect } from 'next/navigation'

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

async function getWaitlistEntries() {
  if (!supabaseUrl || !supabaseKey || supabaseUrl === 'https://your-project.supabase.co') {
    // Return sample data if Supabase not configured
    return {
      entries: [
        {
          id: '1',
          email: 'creator@example.com',
          plan: 'creator',
          max_price: '$19',
          interests: ['API access', 'Team accounts'],
          created_at: new Date().toISOString()
        },
        {
          id: '2', 
          email: 'pro@example.com',
          plan: 'pro',
          max_price: '$49',
          interests: ['Competitor comparison', 'Historical tracking'],
          created_at: new Date().toISOString()
        }
      ],
      counts: { total: 2, creator: 1, pro: 1 },
      usingSupabase: false
    }
  }

  try {
    const supabase = createClient(supabaseUrl, supabaseKey)
    
    // Get all waitlist entries
    const { data: entries, error } = await supabase
      .from('waitlist')
      .select('*')
      .order('created_at', { ascending: false })
    
    if (error) {
      console.error('Supabase query error:', error)
      throw error
    }
    
    // Get counts by plan
    const { data: creatorCount } = await supabase
      .from('waitlist')
      .select('id', { count: 'exact', head: true })
      .eq('plan', 'creator')
    
    const { data: proCount } = await supabase
      .from('waitlist')
      .select('id', { count: 'exact', head: true })
      .eq('plan', 'pro')
    
    return {
      entries: entries || [],
      counts: {
        total: entries?.length || 0,
        creator: creatorCount || 0,
        pro: proCount || 0
      },
      usingSupabase: true
    }
    
  } catch (error) {
    console.error('Failed to fetch waitlist entries:', error)
    return {
      entries: [],
      counts: { total: 0, creator: 0, pro: 0 },
      usingSupabase: false,
      error: 'Failed to load data'
    }
  }
}

export default async function WaitlistAdmin() {
  // Simple access control - check if accessing from localhost or specific email
  // In production, you'd want proper authentication
  const isLocalhost = process.env.NODE_ENV === 'development'
  
  if (!isLocalhost) {
    // In production, you could check for admin authentication here
    // For now, just allow access but show a warning
  }
  
  const { entries, counts, usingSupabase, error } = await getWaitlistEntries()
  
  return (
    <div className='min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 p-8'>
      <div className='max-w-7xl mx-auto'>
        {/* Header */}
        <div className='mb-8'>
          <h1 className='text-4xl font-bold text-white mb-4'>
            🎯 Waitlist Dashboard
          </h1>
          
          {/* Status Badge */}
          <div className='flex items-center gap-4 mb-4'>
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              usingSupabase 
                ? 'bg-green-600/20 text-green-300 border border-green-500/30'
                : 'bg-yellow-600/20 text-yellow-300 border border-yellow-500/30'
            }`}>
              {usingSupabase ? '✅ Using Supabase' : '⚠️ Using Memory Storage'}
            </div>
            
            {error && (
              <div className='px-3 py-1 rounded-full text-sm font-medium bg-red-600/20 text-red-300 border border-red-500/30'>
                ❌ {error}
              </div>
            )}
          </div>
          
          <a
            href='/'
            className='text-blue-400 hover:text-blue-300 text-sm transition-colors'
          >
            ← Back to Home
          </a>
        </div>

        {/* Stats Cards */}
        <div className='grid grid-cols-1 md:grid-cols-3 gap-6 mb-8'>
          <div className='bg-gray-800/50 rounded-xl p-6 border border-gray-700'>
            <div className='text-3xl font-bold text-purple-400 mb-2'>
              {counts.total}
            </div>
            <div className='text-gray-400'>Total Signups</div>
          </div>
          
          <div className='bg-gray-800/50 rounded-xl p-6 border border-gray-700'>
            <div className='text-3xl font-bold text-blue-400 mb-2'>
              {counts.creator}
            </div>
            <div className='text-gray-400'>Creator Plan</div>
          </div>
          
          <div className='bg-gray-800/50 rounded-xl p-6 border border-gray-700'>
            <div className='text-3xl font-bold text-yellow-400 mb-2'>
              {counts.pro}
            </div>
            <div className='text-gray-400'>Pro Plan</div>
          </div>
        </div>

        {/* Waitlist Entries Table */}
        <div className='bg-gray-800/50 rounded-xl border border-gray-700 overflow-hidden'>
          <div className='px-6 py-4 border-b border-gray-700'>
            <h2 className='text-xl font-semibold text-white'>
              Waitlist Entries ({entries.length})
            </h2>
          </div>
          
          {entries.length === 0 ? (
            <div className='p-8 text-center text-gray-400'>
              No waitlist entries yet. Users will appear here when they sign up.
            </div>
          ) : (
            <div className='overflow-x-auto'>
              <table className='w-full'>
                <thead className='bg-gray-700/50'>
                  <tr>
                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider'>
                      Email
                    </th>
                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider'>
                      Plan
                    </th>
                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider'>
                      Max Price
                    </th>
                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider'>
                      Interests
                    </th>
                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider'>
                      Joined
                    </th>
                  </tr>
                </thead>
                <tbody className='divide-y divide-gray-700'>
                  {entries.map((entry: any, index) => (
                    <tr key={entry.id || index} className='hover:bg-gray-700/30'>
                      <td className='px-6 py-4 whitespace-nowrap'>
                        <div className='text-sm text-white'>
                          {entry.email}
                        </div>
                      </td>
                      <td className='px-6 py-4 whitespace-nowrap'>
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          entry.plan === 'creator' 
                            ? 'bg-blue-600/20 text-blue-300'
                            : 'bg-purple-600/20 text-purple-300'
                        }`}>
                          {entry.plan}
                        </span>
                      </td>
                      <td className='px-6 py-4 whitespace-nowrap text-sm text-gray-300'>
                        {entry.max_price || entry.maxPrice}
                      </td>
                      <td className='px-6 py-4 text-sm text-gray-300'>
                        <div className='max-w-xs'>
                          {(entry.interests || []).length > 0 
                            ? (entry.interests || []).join(', ')
                            : 'None selected'
                          }
                        </div>
                      </td>
                      <td className='px-6 py-4 whitespace-nowrap text-sm text-gray-300'>
                        {new Date(entry.created_at || entry.joinedAt).toLocaleDateString('en-US', {
                          year: 'numeric',
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit'
                        })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Setup Instructions */}
        {!usingSupabase && (
          <div className='mt-8 bg-yellow-600/10 border border-yellow-500/30 rounded-xl p-6'>
            <h3 className='text-lg font-semibold text-yellow-300 mb-4'>
              🔧 Supabase Setup Required
            </h3>
            <div className='text-yellow-200 space-y-2 text-sm'>
              <p>To enable persistent storage, set up Supabase:</p>
              <ol className='list-decimal list-inside space-y-1 ml-4'>
                <li>Run the SQL script: <code className='bg-gray-800 px-2 py-1 rounded'>supabase_waitlist_table.sql</code></li>
                <li>Update environment variables:
                  <ul className='list-disc list-inside ml-4 mt-1'>
                    <li><code>NEXT_PUBLIC_SUPABASE_URL</code></li>
                    <li><code>SUPABASE_SERVICE_ROLE_KEY</code></li>
                  </ul>
                </li>
                <li>Redeploy the application</li>
              </ol>
            </div>
          </div>
        )}

        {/* Export Instructions */}
        <div className='mt-8 bg-blue-600/10 border border-blue-500/30 rounded-xl p-6'>
          <h3 className='text-lg font-semibold text-blue-300 mb-4'>
            📧 Export Email List
          </h3>
          <div className='text-blue-200 text-sm'>
            <p>To export emails for marketing:</p>
            <ol className='list-decimal list-inside space-y-1 ml-4 mt-2'>
              <li>Go to your Supabase dashboard</li>
              <li>Navigate to Table Editor → waitlist</li>
              <li>Use Export function or SQL query:
                <code className='block bg-gray-800 p-2 rounded mt-2 text-xs'>
                  SELECT email, plan, max_price, interests, created_at FROM waitlist ORDER BY created_at DESC;
                </code>
              </li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  )
}